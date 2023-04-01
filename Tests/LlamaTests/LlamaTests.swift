import XCTest
@testable import Llama

final class LlamaTests: XCTestCase {
    var llama: Llama!

    override func setUp() async throws {
        let bundlePath = Bundle.module.path(forResource: "gpt4all-lora-quantized", ofType: "bin")!

        var params = LlamaContextParams()
        params.embedding = true
        llama = try Llama(path: bundlePath, contextParams: params)
    }

    func testTokenize() throws {
        XCTAssertEqual(llama.tokenize("Hello World"), [10994, 2787])
        XCTAssertEqual(llama.tokenize("How many letters are there in the English alphabet?"), [5328, 1784, 8721, 526, 727, 297, 278, 4223, 22968, 29973])
        XCTAssertEqual(llama.tokenize("中文測試"), [30275, 30333, 233, 187, 175, 235, 172, 169])
    }

    func testPredict() throws {
        var params = LlamaSampleParams.default
        params.temperature = 0.0001
        let result = try llama.predict("Neil Armstrong: That's one small step for a man,", predicts: 64, params: params)
        XCTAssertNotNil(result)
        XCTAssertTrue(result.contains("one giant leap for mankind."))

        let result2 = try llama.predict("Steve Job: Your time is limited, so don't waste it", predicts: 64, params: params)
        XCTAssertNotNil(result2)
        XCTAssertTrue(result2.contains("living someone else's life. Don't be trapped by dogma - which is living with the results of other people's thinking. Don't let the noise of other's opinions drown out your own inner voice. And most important, have the courage to follow your heart and intuition."))
    }

    func testEmbedding() throws {
        var params = LlamaSampleParams.default
        params.temperature = 0.0001
        let result = try llama.embedding("London bridge is falling down")
        XCTAssertGreaterThan(result.count, 0)
        XCTAssertEqual(result.count, 4096)

        let result2 = try llama.embedding("Falling down, falling down")
        XCTAssertGreaterThan(result2.count, 0)
        XCTAssertEqual(result2.count, 4096)
    }
}
