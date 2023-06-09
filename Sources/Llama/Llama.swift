import Foundation
import CLlama

private typealias _LlamaProgressCallback = (_ progress: Float, _ userData: UnsafeMutableRawPointer?) -> Void

public typealias LlamaProgressCallback = (_ progress: Float, _ llama: Llama) -> Void

public struct LlamaContextParams {
    public var context: Int32 = 512    // text context
    public var parts: Int32 = -1   // -1 for default
    public var seed: Int32 = 0      // RNG seed, 0 for random
    public var numberOFThread: Int32 = 4

    public var f16Kv = true         // use fp16 for KV cache
    public var logitsAll = false    // the llama_eval() call computes all logits, not just the last one
    public var vocabOnly = false    // only load the vocabulary, no weights
    public var useMlock = false     // force system to keep model in RAM
    public var embedding = false    // embedding mode only

    public static let `default` = LlamaContextParams()

    public init(context: Int32 = 512, parts: Int32 = -1, seed: Int32 = 0, numberOFThread: Int32 = 4, f16Kv: Bool = true, logitsAll: Bool = false, vocabOnly: Bool = false, useMlock: Bool = false, embedding: Bool = false) {
        self.context = context
        self.parts = parts
        self.seed = seed
        self.numberOFThread = numberOFThread
        self.f16Kv = f16Kv
        self.logitsAll = logitsAll
        self.vocabOnly = vocabOnly
        self.useMlock = useMlock
        self.embedding = embedding
    }
}

public struct LlamaSampleParams {
    public var topK: Int32
    public var topP: Float
    public var temperature: Float
    public var repeatLastN: Int32
    public var repeatPenalty: Float
    public var batchSize: Int32

    public static let `default` = LlamaSampleParams(
        topK: 40,
        topP: 0.95,
        temperature: 0.8,
        repeatLastN: 64,
        repeatPenalty: 1.1,
        batchSize: 8
    )

    public init(topK: Int32 = 40, topP: Float = 0.95, temperature: Float = 0.8, repeatLastN: Int32 = 64, repeatPenalty: Float = 1.1, batchSize: Int32 = 8) {
        self.topK = topK
        self.topP = topP
        self.temperature = temperature
        self.repeatLastN = repeatLastN
        self.repeatPenalty = repeatPenalty
        self.batchSize = batchSize
    }
}

public enum LlamaError: Error {
    case modelNotFound(String)
    case inputTooLong
    case failedToEval
}

public class Llama {
    private let context: OpaquePointer?
    private var contextParams: LlamaContextParams

    public init(path: String, contextParams: LlamaContextParams = .default) throws {
        self.contextParams = contextParams
        var params = llama_context_default_params()
        params.n_ctx = contextParams.context
        params.n_parts = contextParams.parts
        params.seed = contextParams.seed
        params.f16_kv = contextParams.f16Kv
        params.logits_all = contextParams.logitsAll
        params.vocab_only = contextParams.vocabOnly
        params.use_mlock = contextParams.useMlock
        params.embedding = contextParams.embedding

        if !FileManager.default.fileExists(atPath: path) {
            throw LlamaError.modelNotFound(path)
        }

        context = llama_init_from_file(path, params)
    }

    public func predict(_ input: String, count: Int = 128, params: LlamaSampleParams = .default) throws -> String {
        // Add a space in front of the first character to match OG llama tokenizer behavior
        let input = " " + input
        // tokenize the prompt
        let inputs = tokenize(input, addBos: true)

        var outputs = Array<llama_token>()
        var outputStrings = [String]()
        var consumed = 0
        var remain = count
        var nPast = Int32(0)

        if inputs.count > contextParams.context - 4 {
            throw LlamaError.inputTooLong
        }

        var lastNTokens = Array<llama_token>()
        while remain != 0 {
            if outputs.count > 0 {
                if llama_eval(context, outputs, Int32(outputs.count), nPast, contextParams.numberOFThread) != 0 {
                    throw LlamaError.failedToEval
                }
            }

            nPast += Int32(outputs.count)
            outputs.removeAll()

            if inputs.count <= consumed {
                let skipped = Int(contextParams.context - params.repeatLastN)
                let lastNTokensSkipped = lastNTokens.count > skipped ? Array(lastNTokens.suffix(from: skipped)) : lastNTokens
                let id = llama_sample_top_p_top_k(
                    context,
                    lastNTokensSkipped,
                    Int32(lastNTokensSkipped.count),
                    params.topK,
                    params.topP,
                    params.temperature,
                    params.repeatPenalty
                )
                lastNTokens.removeFirst()
                lastNTokens.append(id)

                outputs.append(id)
                remain -= 1
            } else {
                while inputs.count > consumed {
                    outputs.append(inputs[consumed])
                    if lastNTokens.count > 0 {
                        lastNTokens.removeFirst()
                    }
                    lastNTokens.append(inputs[consumed])
                    consumed += 1
                    if outputs.count >= params.batchSize {
                        break
                    }
                }
            }

            for outputToken in outputs {
                if let str = llama_token_to_str(context, outputToken) {
                    outputStrings.append(String(cString: str))
                }
            }

            if outputs.last == llama_token_eos() {
                break
            }
        }

        return outputStrings.joined()
    }

    public func embeddings(_ input: String) throws -> [Float] {
        // Add a space in front of the first character to match OG llama tokenizer behavior
        let input = " " + input

        // tokenize the prompt
        let inputs = tokenize(input, addBos: true)

        guard inputs.count > 0 else {
            return []
        }

        if llama_eval(context, inputs, Int32(inputs.count), Int32(0), contextParams.numberOFThread) != 0 {
            throw LlamaError.failedToEval
        }

        let embeddingsCount = Int(llama_n_embd(context))
        guard let embeddings = llama_get_embeddings(context) else {
            return []
        }
        return Array(UnsafeBufferPointer(start: embeddings, count: embeddingsCount))
    }

    deinit {
        llama_free(context)
    }

    // MARK: Internal

    func tokenize(_ input: String, addBos: Bool = false) -> [llama_token] {
        if input.count == 0 {
            return []
        }

        var embeddings: [llama_token] = Array<llama_token>(repeating: llama_token(), count: input.utf8.count)
        let n = llama_tokenize(context, input, &embeddings, Int32(input.utf8.count), addBos)
        assert(n >= 0)
        embeddings.removeSubrange(Int(n)..<embeddings.count)
        return embeddings
    }
}
