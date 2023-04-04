import XCTest
@testable import Llama

final class LlamaTests: XCTestCase {
    var llama: Llama!

    override func setUp() async throws {
        let bundlePath = Bundle.module.path(forResource: "gpt4all-lora-quantized", ofType: "bin")
        try XCTSkipIf(bundlePath == nil, "Tests/LlamaTests/Resources/gpt4all-lora-quantized.bin not found, have you download the file?")

        var params = LlamaContextParams()
        params.embedding = true
        llama = try Llama(path: bundlePath!, contextParams: params)
    }

    func testTokenize() throws {
        XCTAssertEqual(llama.tokenize("Hello World"), [10994, 2787])
        XCTAssertEqual(llama.tokenize("How many letters are there in the English alphabet?"), [5328, 1784, 8721, 526, 727, 297, 278, 4223, 22968, 29973])
        XCTAssertEqual(llama.tokenize("中文測試"), [30275, 30333, 233, 187, 175, 235, 172, 169])
    }

    func testPredict() throws {
        var params = LlamaSampleParams.default
        params.temperature = 0.0001
        let result = try llama.predict("Neil Armstrong: That's one small step for a man,", count: 64, params: params)
        XCTAssertNotNil(result)
        XCTAssertTrue(result.contains("one giant leap for mankind."))

        let result2 = try llama.predict("Steve Job: Your time is limited, so don't waste it", count: 64, params: params)
        XCTAssertNotNil(result2)
        XCTAssertTrue(result2.contains("living someone else's life."))
    }

    func testEmbedding() throws {
        var params = LlamaSampleParams.default
        params.temperature = 0.0001
        let result = try llama.embeddings("London bridge is falling down")
        XCTAssertGreaterThan(result.count, 0)
        XCTAssertEqual(result.count, 4096)

        let result2 = try llama.embeddings("Falling down, falling down")
        XCTAssertGreaterThan(result2.count, 0)
        XCTAssertEqual(result2.count, 4096)
    }

    func testQuestionAnswering() throws {
        let prompt = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

The device was announced and unveiled on January 27, 2010, by Apple founder Steve Jobs at an Apple press event. On April 3, 2010, the Wi-Fi variant of the device was released in the United States, followed by the release of the "Wi-Fi + 3G" variant on April 30. On May 28, 2010, it was released in Australia, Canada, France, Japan, Italy, Germany, Spain, Switzerland and the United Kingdom.

Question: When is iPad first released in United Kingdom?

Helpful Answer:
"""

        let result = try llama.predict(prompt, count: 2024)
        print(result)

        let prompt2 = """
Question: When is iPad first released in United Kingdom?

Helpful Answer:
"""
        let result2 = try llama.predict(prompt2, count: 2024)
        print(result2)
    }
}
