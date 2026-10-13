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
 */
object JsonInject {

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
