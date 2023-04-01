import Foundation
import CLlama

private typealias _LlamaProgressCallback = (_ progress: Float, _ userData: UnsafeMutableRawPointer?) -> Void

public typealias LlamaProgressCallback = (_ progress: Float, _ llama: Llama) -> Void

public struct LlamaContextParams {
    public var nCtx: Int32 = 256    // text context
    public var nParts: Int32 = -1   // -1 for default
    public var seed: Int32 = 0      // RNG seed, 0 for random

    public var f16Kv = true         // use fp16 for KV cache
    public var logitsAll = false    // the llama_eval() call computes all logits, not just the last one
    public var vocabOnly = false    // only load the vocabulary, no weights
    public var useMlock = false     // force system to keep model in RAM
    public var embedding = false    // embedding mode only

    // called with a progress value between 0 and 1, pass nil to disable
    public var progressCallback: LlamaProgressCallback?
    // context pointer passed to the progress callback
    public var progressCallbackUserData: UnsafeMutableRawPointer?
}

public enum LlamaError: Error {
    case modelNotFound(String)
}

public class Llama {
    private let llamaContext: OpaquePointer?

    public init(path: String, contextParams: LlamaContextParams) throws {
        var params = llama_context_params()
        params.n_ctx = contextParams.nCtx
        params.n_parts = contextParams.nParts
        params.seed = contextParams.seed
        params.f16_kv = contextParams.f16Kv
        params.logits_all = contextParams.logitsAll
        params.vocab_only = contextParams.vocabOnly
        params.use_mlock = contextParams.useMlock
        params.embedding = contextParams.embedding

        if !FileManager.default.fileExists(atPath: path) {
            throw LlamaError.modelNotFound(path)
        }

        print("llama_init_from_file(\(path)")
        llamaContext = llama_init_from_file(path, params)
    }

    func tokenize(_ input: String) -> [llama_token] {
        var embeddings: [llama_token] = Array<llama_token>(repeating: llama_token(), count: input.utf8.count)
        let n = llama_tokenize(llamaContext, input, &embeddings, Int32(input.utf8.count), false)
        assert(n >= 0)
        embeddings.removeSubrange(Int(n)..<embeddings.count)
        return embeddings
    }

    deinit {
        llama_free(llamaContext)
    }
}
