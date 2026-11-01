package org.edgeslm.runner

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.util.Log

/**
 * UI を持たない入口。
 *
 * 存在理由は2つだけ（基本設計③ 3章）:
 *  ① インストール直後の「stopped 状態」から抜ける入口を用意する
 *  ② **Android 12+ はバックグラウンドからの Foreground Service 起動を制限する**ため、
 *     フォアグラウンド文脈（＝アクティビティ）から Service を起動する
 *
 * 起動されたらすぐ Service を開始して finish する。画面は出ない。
 *
 * 使い方:
 *   adb shell am start -n org.edgeslm.runner/.RunnerActivity \
 *       -a org.edgeslm.runner.RUN --es job jobs/job.json
 *
 * （2026-10-18 追記）学習用の説明:
 *   - 普段は scripts/run_runner.sh run jobs/<job>.json が、上の am start を代わりに打つ（android_runner/README.md）
 *   - 入力: Intent の extras（job = files からの相対パス・overrides = 上書き用の JSON。任意）
 *   - 出力: RunnerService の起動（画面は出さない）と logcat への行（externalFilesDir=…・subdirs ready: …）
 *   - 段の並び: ① 置き場のパスをログに出す → ② 置き場の下のフォルダを作る → ③ job が無ければ版だけ出して終わる
 *               → ④ Service に job と overrides を渡して起動し、すぐ finish
 */
class RunnerActivity : Activity() {

    /** 起動されたときに 1 回だけ走る（上の ①〜④）。終わりに必ず finish するので画面は残らない */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // ① 置き場のパス
        // 外部ファイルディレクトリの実パスを必ずログに出す。
        // ホスト側のハーネスはこのパスに push する（基本設計③ リスク#5）。
        val base = getExternalFilesDir(null)
        Log.i(TAG, "externalFilesDir=${base?.absolutePath}")

        // ② 置き場の下のフォルダ（models・models/ops・jobs・inputs・out）を作る
        // ★サブディレクトリは **アプリ自身が作る**。
        // adb shell の mkdir で作ると所有者が shell になり、
        // アプリ側から入れず「job file not found」になる（2026-09-12 に実際に踏んだ）。
        // files 直下はアプリ所有かつ group=ext_data_rw なので、
        // アプリが作ったディレクトリには adb（shell）からも push できる。
        base?.let { b ->
            listOf("models", "models/ops", "jobs", "inputs", "out").forEach { sub ->
                val d = java.io.File(b, sub)
                if (!d.isDirectory && !d.mkdirs()) Log.w(TAG, "mkdir failed: ${d.absolutePath}")
            }
            Log.i(TAG, "subdirs ready: ${b.list()?.joinToString(",")}")
        }

        // ③ job の指定があるか
        val job = intent?.getStringExtra(EXTRA_JOB)
        if (job.isNullOrBlank()) {
            // ジョブ指定なしで起動された場合（ランチャーからのタップ等）は
            // パスだけ報告して終了する。これも「疎通確認」として有用。
            Log.i(TAG, "no job specified; nothing to run. versions=${NativeBridge.nativeVersions()}")
            finish()
            return
        }

        // ④ Foreground Service として起動する（Android 12 以降は画面のある文脈から起こす必要がある。上の説明の②）
        val svc = Intent(this, RunnerService::class.java).apply {
            action = ACTION_RUN
            putExtra(EXTRA_JOB, job)
            // job.json の一部を extras で上書きできるようにしておく（基本設計③ 5.1節）
            intent.getStringExtra(EXTRA_OVERRIDES)?.let { putExtra(EXTRA_OVERRIDES, it) }
        }
        Log.i(TAG, "starting RunnerService with job=$job")
        startForegroundService(svc)
        finish()
    }

    // Intent の名前（action・extras のキー）とログの札。RunnerService も同じ名前を使う
    companion object {
        const val TAG = "EdgeSLMRunner"
        const val ACTION_RUN = "org.edgeslm.runner.RUN"
        const val EXTRA_JOB = "job"
        const val EXTRA_OVERRIDES = "overrides"
    }
}
