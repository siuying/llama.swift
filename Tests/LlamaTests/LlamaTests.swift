import XCTest
@testable import Llama

final class LlamaTests: XCTestCase {
    var llama: Llama!

    override func setUp() async throws {
        let bundlePath = Bundle.module.path(forResource: "ggml-vocab", ofType: "bin")!
        llama = try Llama(path: bundlePath, contextParams: LlamaContextParams())
    }

    func testTokenize() throws {
        XCTAssertEqual(llama.tokenize("Hello World"), [10994, 2787])
        XCTAssertEqual(llama.tokenize("How many letters are there in the English alphabet?"), [5328, 1784, 8721, 526, 727, 297, 278, 4223, 22968, 29973])
        XCTAssertEqual(llama.tokenize("中文測試"), [30275, 30333, 233, 187, 175, 235, 172, 169])
    }
}
