package org.edgeslm.runner

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.concurrent.thread

/**
 * 推論を実行する Foreground Service。UI は持たない。
 *
 * 責務は最小限に絞る（基本設計③ 6.1節）:
 *   - ジョブ仕様ファイルを読む
 *   - JNI を呼ぶ
 *   - 返ってきた JSON をファイルへ書く
 *
 * **推論と計測のロジックは持たない。** それらは C++ 側にある。
 * 理由は ①最終目標が「C++ 推論サービス」であること
 *        ②Kotlin を触ると APK の再ビルドが必要になること。
 *
 * **端末の状態監視（温度・クロック・dma-buf）も持たない。**
 * それはホスト側のハーネスが adb 経由で並行して行う（基本設計① 8章）。
 * 責務を分けることで、計測項目を増やしてもこのアプリを作り直さずに済む。
 *
 * （2026-10-18 追記）学習用の説明:
 *   - 入力: RunnerActivity から渡る job の相対パス（例 jobs/p3_qwen24.json）と overrides（任意）
 *   - 出力: files/out/<種類>_<yyyyMMdd_HHmmss>.json と files/out/last_<種類>.json（種類は result・describe・error）。
 *           PC 側（scripts/run_runner.sh・app/pipeline/device_accuracy.py など）は last_result.json を adb pull して読む
 *   - 流れ: onStartCommand（前面に出す → WakeLock → 作業用スレッドを立てる）→ runJob（読む → 注入 → 開く → 実行 → 書く）
 *           → stopSelfSafely（WakeLock を返して止まる）。1 回の起動でジョブ 1 件
 */
class RunnerService : Service() {

    // handle: C++ の Handle の番地（0 = 開いていない）。作業用スレッドが書き、onDestroy（主スレッド）が読むので @Volatile
    @Volatile private var handle: Long = 0L
    private var worker: Thread? = null
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    /**
     * Service が起こされたとき（startForegroundService の後）に呼ばれる。
     * 通知を出して前面に出し、WakeLock を取り、ジョブの実行は別スレッド（edgeslm-job）に任せてすぐ戻る。
     * 主スレッドで推論すると、時間がかかったときにアプリが応答なしで落とされるため（推定）。
     * 戻り値 START_NOT_STICKY: 落とされても自動では起き直さない（ジョブを勝手にやり直さない）
     */
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // ① すぐに前面に出す（startForegroundService で起こされた Service は、決まった時間の中でこれを呼ぶ必要がある）
        startForeground(NOTIFICATION_ID, buildNotification("running"))

        val jobRelPath = intent?.getStringExtra(RunnerActivity.EXTRA_JOB)
        val overrides = intent?.getStringExtra(RunnerActivity.EXTRA_OVERRIDES)
        if (jobRelPath.isNullOrBlank()) {
            Log.e(TAG, "no job path given")
            stopSelfSafely()
            return START_NOT_STICKY
        }

        // 画面を消したまま連続実行したい（発熱要因を減らすため）ので PARTIAL_WAKE_LOCK。
        // 取得に失敗しても**ジョブは続行する**。WakeLock は「あると望ましい」ものであって、
        // これが原因で検証全体が止まるのは割に合わないため。
        try {
            val pm = getSystemService(POWER_SERVICE) as PowerManager
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$TAG:job").apply {
                setReferenceCounted(false)
                acquire(WAKELOCK_TIMEOUT_MS)
            }
        } catch (t: Throwable) {
            Log.w(TAG, "wake lock unavailable; continuing without it: ${t.message}")
            wakeLock = null
        }

        // ② 作業用スレッドでジョブを走らせる
        //    （2026-10-18 注）前のジョブが終わる前に次の起動が来ると、worker と handle が上書きされる。
        //    ジョブは 1 件ずつ送る前提（run_runner.sh は結果を待ってから次を送る、と推定）
        worker = thread(name = "edgeslm-job") { runJob(jobRelPath, overrides) }
        return START_NOT_STICKY
    }

    /**
     * ジョブ 1 件を実行して結果を書く（作業用スレッドで走る）。
     * ① 置き場と job のファイルを確かめる → ② runtime のパスを注入する（JsonInject）
     * → ③ nativeCreate でセッションを開き、describe をログに出す → 開けなければ describe を書いて終わる
     * → ④ nativeRunJob の結果を書く → ⑤ 例外でも finally でハンドルを捨てて Service を止める
     *
     * @param jobRelPath 置き場（getExternalFilesDir）からの相対パス
     * @param overrides  上書き用の JSON（無ければ null）
     */
    private fun runJob(jobRelPath: String, overrides: String?) {
        val base = getExternalFilesDir(null)
        if (base == null) {
            Log.e(TAG, "external files dir unavailable")
            stopSelfSafely(); return
        }
        Log.i(TAG, "externalFilesDir=${base.absolutePath}")
        Log.i(TAG, "versions=${NativeBridge.nativeVersions()}")

        // ① job のファイルが無ければ、理由をログに出して止まる（結果のファイルは書かない）
        val jobFile = File(base, jobRelPath)
        if (!jobFile.isFile) {
            Log.e(TAG, "job file not found: ${jobFile.absolutePath}")
            stopSelfSafely(); return
        }

        try {
            var jobJson = jobFile.readText()

            // ネイティブ側が相対パスを解決できるよう、基準ディレクトリと
            // ネイティブライブラリディレクトリを注入する。
            // 後者は **FastRPC が libQnnHtp.so の実ファイルパスを要求する**ため必須（C-6）。
            val injected = JsonInject.addRuntimePaths(
                jobJson,
                baseDir = base.absolutePath,
                nativeLibDir = applicationInfo.nativeLibraryDir,
                overridesJson = overrides
            )

            Log.i(TAG, "job=${jobFile.name} nativeLibDir=${applicationInfo.nativeLibraryDir}")

            // ③ 開く。この JSON が設定（model・ep・runtime）とジョブ（task など）を兼ねる
            handle = NativeBridge.nativeCreate(injected)
            val describe = NativeBridge.nativeDescribe(handle)
            Log.i(TAG, "describe=$describe")

            if (handle == 0L) {
                writeResult(base, "describe", describe)
                stopSelfSafely(); return
            }

            // ④ 実行（終わるまで戻らない）。ログには先頭 2000 文字だけ出し、全文はファイルに書く
            val result = NativeBridge.nativeRunJob(handle, injected)
            Log.i(TAG, "result=${result.take(2000)}")
            writeResult(base, "result", result)
        } catch (t: Throwable) {
            // （2026-10-18 注）下の行は例外の文言をそのまま JSON に埋め込むので、文言に " が入ると JSON として読めない
            Log.e(TAG, "job failed", t)
            writeResult(base, "error", """{"error":"${t.javaClass.simpleName}: ${t.message}"}""")
        } finally {
            // ⑤ 成功でも失敗でも、開いたセッションを捨てて Service を止める
            if (handle != 0L) {
                NativeBridge.nativeDestroy(handle)
                handle = 0L
            }
            stopSelfSafely()
        }
    }

    /** 結果はファイルとログの両方に出す。ホストは pull でもログ回収でも取れる。 */
    private fun writeResult(base: File, kind: String, json: String) {
        try {
            val outDir = File(base, "out").apply { mkdirs() }
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            File(outDir, "${kind}_$stamp.json").writeText(json)
            // 直近の結果は固定名でも残す（ハーネスが取りに行きやすいように）
            File(outDir, "last_$kind.json").writeText(json)
        } catch (t: Throwable) {
            Log.e(TAG, "failed to write $kind", t)
        }
    }

    /**
     * WakeLock を返し、通知を消して Service を止める。どの終わり方（正常・失敗・job 無し）でも最後にここを通る。
     * WakeLock の解放の失敗は握りつぶす（止めることを優先する）
     */
    private fun stopSelfSafely() {
        try { wakeLock?.let { if (it.isHeld) it.release() } } catch (_: Throwable) {}
        wakeLock = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    /**
     * Service が止められるとき。実行中のジョブがあれば C++ に「止めて」と頼む（nativeCancel。handle が 0 なら何もしない）。
     * （2026-10-18 注）作業用スレッドが同時に nativeDestroy していると、捨てた後の番地に触る恐れがある（推定。コードは直していない）
     */
    override fun onDestroy() {
        NativeBridge.nativeCancel(handle)
        super.onDestroy()
    }

    /** 前面の Service に必要な通知を作る（重要度 LOW。音は鳴らない）。text は通知に出す短い状態（"running"） */
    private fun buildNotification(text: String): Notification {
        val nm = getSystemService(NotificationManager::class.java)
        val ch = NotificationChannel(CHANNEL_ID, "EdgeSLM Runner", NotificationManager.IMPORTANCE_LOW)
        nm.createNotificationChannel(ch)
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("EdgeSLM Runner")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setOngoing(true)
            .build()
    }

    companion object {
        const val TAG = "EdgeSLMRunner"
        private const val CHANNEL_ID = "edgeslm_runner"
        private const val NOTIFICATION_ID = 1
        // WakeLock は 30 分で自動的に切れる（解放し忘れても電池を食い続けないための上限）
        // soak（最大5分）でも切れないよう余裕を持たせる
        private const val WAKELOCK_TIMEOUT_MS = 30L * 60L * 1000L
    }
}
