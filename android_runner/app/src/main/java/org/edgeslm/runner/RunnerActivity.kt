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
 */
class RunnerActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 外部ファイルディレクトリの実パスを必ずログに出す。
        // ホスト側のハーネスはこのパスに push する（基本設計③ リスク#5）。
        val base = getExternalFilesDir(null)
        Log.i(TAG, "externalFilesDir=${base?.absolutePath}")

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

        val job = intent?.getStringExtra(EXTRA_JOB)
        if (job.isNullOrBlank()) {
            // ジョブ指定なしで起動された場合（ランチャーからのタップ等）は
            // パスだけ報告して終了する。これも「疎通確認」として有用。
            Log.i(TAG, "no job specified; nothing to run. versions=${NativeBridge.nativeVersions()}")
            finish()
            return
        }

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

    companion object {
        const val TAG = "EdgeSLMRunner"
        const val ACTION_RUN = "org.edgeslm.runner.RUN"
        const val EXTRA_JOB = "job"
        const val EXTRA_OVERRIDES = "overrides"
    }
}
