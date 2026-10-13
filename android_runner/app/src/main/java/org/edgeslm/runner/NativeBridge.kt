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
 */
object NativeBridge {

    init {
        System.loadLibrary("edgeslm_runner")
    }

    /**
     * ORT / QNN のセッションを生成する。
     * @param configJson モデルパス・EP オプション・セッション設定を含む JSON
     * @return セッションハンドル。0 なら失敗（理由は describe/エラー JSON で取得）
     */
    external fun nativeCreate(configJson: String): Long

    /**
     * バックエンド名・グラフ配置・入出力情報を JSON で返す。
     * セッション生成に失敗した場合もエラー内容を JSON で返す。
     */
    external fun nativeDescribe(handle: Long): String

    /**
     * ジョブを実行し、結果を JSON で返す。
     * task（probe / bench / op_check / soak / generate）は jobJson の中で指定する。
     */
    external fun nativeRunJob(handle: Long, jobJson: String): String

    /** soak / generate の中断要求 */
    external fun nativeCancel(handle: Long): Boolean

    /** セッションを破棄する */
    external fun nativeDestroy(handle: Long)

    /** ORT / QNN の版を JSON で返す（FR-12 の記録用。セッション不要） */
    external fun nativeVersions(): String
}
