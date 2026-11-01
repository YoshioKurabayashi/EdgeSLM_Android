package org.edgeslm.runner

import org.json.JSONObject

/**
 * ジョブ仕様 JSON に実行時パスを注入し、extras による上書きを適用する。
 *
 * ここは「ロジック」ではなくグルーである。ネイティブ側は端末固有のパスを知りようがないため、
 * Kotlin 側でしか取得できない2つのパスだけを渡す。
 *
 *  - base_dir       : getExternalFilesDir()。モデル・入力・結果の基準
 *  - native_lib_dir : applicationInfo.nativeLibraryDir。
 *                     **FastRPC が libQnnHtp.so の実ファイルパスを要求する**ため必須（C-6）
 *
 * （2026-10-18 追記）学習用の説明:
 *  - どこで使うか: RunnerService.runJob が job.json を読んだ直後に 1 回呼ぶ（基本設計③ 5.1節の「extras で上書き」もここ）
 *  - 入力と出力: job の JSON 文字列・2 つのパス・上書き用の JSON（無ければ null）→ 注入と上書きを済ませた JSON 文字列
 *  - C++ 側（Session::resolve_path）は runtime.base_dir を基準に、job の相対パス（models/…・inputs/…・out/…）を絶対パスにする
 */
object JsonInject {

    /**
     * job の "runtime" に base_dir と native_lib_dir を書き込み、続けて上書き（overridesJson）を当てる。
     *
     * @param jobJson       jobs/ に置いた job の .json の中身
     * @param baseDir       アプリの外部ファイルの置き場（/sdcard/Android/data/org.edgeslm.runner/files）
     * @param nativeLibDir  APK から展開された .so の置き場（QNN のライブラリがある）
     * @param overridesJson am start の --es overrides で渡した JSON。空なら何もしない
     * @return 注入済みの JSON 文字列。この文字列が nativeCreate（設定）と nativeRunJob（ジョブ）の両方に渡る
     *
     * 順番の注意: 上書きは注入の後に当たるので、overrides に runtime.base_dir を書くと注入した値を上書きできてしまう
     * （2026-10-18 注。推定ではなくコードの順から言えること。使った記録は見当たらない）
     */
    fun addRuntimePaths(
        jobJson: String,
        baseDir: String,
        nativeLibDir: String,
        overridesJson: String?
    ): String {
        val job = JSONObject(jobJson)

        val runtime = job.optJSONObject("runtime") ?: JSONObject()
        runtime.put("base_dir", baseDir)
        runtime.put("native_lib_dir", nativeLibDir)
        job.put("runtime", runtime)

        if (!overridesJson.isNullOrBlank()) {
            mergeInto(job, JSONObject(overridesJson))
        }
        return job.toString()
    }

    // （2026-10-18 追記）patch の各キーについて、両方がオブジェクトなら中へ降りて同じことをし、
    // それ以外（数・文字列・配列・片方だけオブジェクト）は patch の値で丸ごと置き換える。配列は要素ごとには混ぜない
    /** 浅いマージではなくオブジェクトは再帰的に上書きする */
    private fun mergeInto(target: JSONObject, patch: JSONObject) {
        for (key in patch.keys()) {
            val pv = patch.get(key)
            val tv = target.opt(key)
            if (pv is JSONObject && tv is JSONObject) {
                mergeInto(tv, pv)
            } else {
                target.put(key, pv)
            }
        }
    }
}
