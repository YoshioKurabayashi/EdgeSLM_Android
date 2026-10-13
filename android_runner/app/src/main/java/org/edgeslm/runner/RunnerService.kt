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
 */
class RunnerService : Service() {

    @Volatile private var handle: Long = 0L
    private var worker: Thread? = null
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
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

        worker = thread(name = "edgeslm-job") { runJob(jobRelPath, overrides) }
        return START_NOT_STICKY
    }

    private fun runJob(jobRelPath: String, overrides: String?) {
        val base = getExternalFilesDir(null)
        if (base == null) {
            Log.e(TAG, "external files dir unavailable")
            stopSelfSafely(); return
        }
        Log.i(TAG, "externalFilesDir=${base.absolutePath}")
        Log.i(TAG, "versions=${NativeBridge.nativeVersions()}")

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

            handle = NativeBridge.nativeCreate(injected)
            val describe = NativeBridge.nativeDescribe(handle)
            Log.i(TAG, "describe=$describe")

            if (handle == 0L) {
                writeResult(base, "describe", describe)
                stopSelfSafely(); return
            }

            val result = NativeBridge.nativeRunJob(handle, injected)
            Log.i(TAG, "result=${result.take(2000)}")
            writeResult(base, "result", result)
        } catch (t: Throwable) {
            Log.e(TAG, "job failed", t)
            writeResult(base, "error", """{"error":"${t.javaClass.simpleName}: ${t.message}"}""")
        } finally {
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

    private fun stopSelfSafely() {
        try { wakeLock?.let { if (it.isHeld) it.release() } } catch (_: Throwable) {}
        wakeLock = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        NativeBridge.nativeCancel(handle)
        super.onDestroy()
    }

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
        // soak（最大5分）でも切れないよう余裕を持たせる
        private const val WAKELOCK_TIMEOUT_MS = 30L * 60L * 1000L
    }
}
