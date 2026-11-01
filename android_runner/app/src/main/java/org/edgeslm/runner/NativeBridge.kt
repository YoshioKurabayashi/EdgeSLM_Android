package org.edgeslm.runner

/**
 * JNI 境界。
 *
 * 設計方針（基本設計③ 6.2節）:
 *   引数・戻り値を **JSON 文字列に統一**している。パラメータを1つ増やすたびに
 *   シグネチャが変わると、そのたびに APK の再ビルドが必要になるため。
 *   JSON なら変更が C++ 側に閉じ、この Kotlin ファイルは触らずに済む。
 *
 *   ビルドは（JVM の通信制約により）ユーザー実行になるため、
 *   再ビルドを要する変更を減らすことが設計上の最優先事項である。
 *
 * （2026-10-18 追記）学習用の説明:
 *   - 何のための部品か: Kotlin から C++（edgeslm_jni.cpp → runner/session.cpp・runner/job.cpp）を呼ぶための宣言だけを置く
 *   - 呼ぶ順（RunnerService.runJob）: nativeVersions → nativeCreate → nativeDescribe → nativeRunJob → nativeDestroy
 *   - ハンドル（Long）は C++ の Handle（Session と取り消しの印）の番地。0 は「開けなかった」
 *   - System.loadLibrary の名前 edgeslm_runner は CMakeLists.txt の add_library の名前と一致させる（推定: ずれると起動時に落ちる）
 */
object NativeBridge {

    init {
        // 初めてこの object に触れたとき、libedgeslm_runner.so を読み込む（APK の jniLibs から）
        System.loadLibrary("edgeslm_runner")
    }

    /**
     * ORT / QNN のセッションを生成する。
     * @param configJson モデルパス・EP オプション・セッション設定を含む JSON
     * @return セッションハンドル。0 なら失敗（理由は describe/エラー JSON で取得）
     */
    // ★2026-10-18 訂正: 旧「理由は describe/エラー JSON で取得」。この版の edgeslm_jni.cpp は、開けなかった理由を
    //   logcat（session.open failed: …）にだけ出し、0 を返す。nativeDescribe(0) は「handle が null」としか返さない。
    //   （補足）RunnerService は job の JSON（runtime を注入済み）をそのまま設定として渡す
    external fun nativeCreate(configJson: String): Long

    /**
     * バックエンド名・グラフ配置・入出力情報を JSON で返す。
     * セッション生成に失敗した場合もエラー内容を JSON で返す。
     */
    // ★2026-10-18 訂正: 旧「セッション生成に失敗した場合もエラー内容を JSON で返す」。返るのは
    //   {"ok": false, "error": "describe: session handle is null (open failed; see logcat)"} という決まった文言で、失敗の中身は logcat にある
    external fun nativeDescribe(handle: Long): String

    /**
     * ジョブを実行し、結果を JSON で返す。
     * task（probe / bench / op_check / soak / generate）は jobJson の中で指定する。
     * ★2026-10-18 訂正: 旧「task（probe / bench / op_check / soak / generate）」。この版の C++（job.cpp の
     *   is_supported_task）が実行するのは probe / bench / op_check / infer の 4 つで、soak・generate は「未実装」の
     *   エラーを返す。終わるまで戻らない（同期）ので、RunnerService は別スレッドから呼ぶ。
     */
    external fun nativeRunJob(handle: Long, jobJson: String): String

    /**
     * soak / generate の中断要求
     * ★2026-10-18 訂正: 旧「soak / generate の中断要求」。この版では op_check・infer・bench の反復のループが
     *   取り消しの印を見て止まる（soak・generate は未実装）。印を立てるだけで、止まるのは次にループが印を見たとき。
     *   呼ぶのは RunnerService.onDestroy だけ。
     */
    external fun nativeCancel(handle: Long): Boolean

    /** セッションを破棄する */
    // （2026-10-18 補足）C++ の Handle を delete する。0 なら何もしない。呼んだ後のハンドルは使えない
    external fun nativeDestroy(handle: Long)

    /** ORT / QNN の版を JSON で返す（FR-12 の記録用。セッション不要） */
    external fun nativeVersions(): String
}
