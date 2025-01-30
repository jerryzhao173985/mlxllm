import Foundation
import Hub
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import MLXRandom
import MarkdownUI
import Metal
import SwiftUI
import Tokenizers

//public let typeRegistry = LLMModelFactory.shared.typeRegistry
//LLMModelFactory.shared.modelRegistry.configuration(

struct LLMUserInputProcessor: UserInputProcessor {

    let tokenizer: Tokenizer
    let configuration: ModelConfiguration

    internal init(tokenizer: any Tokenizer, configuration: ModelConfiguration) {
        self.tokenizer = tokenizer
        self.configuration = configuration
    }

    func prepare(input: UserInput) throws -> LMInput {
        do {
            let messages = input.prompt.asMessages()
            let promptTokens = try tokenizer.applyChatTemplate(messages: messages)
            return LMInput(tokens: MLXArray(promptTokens))
        } catch {
            // #150 -- it might be a TokenizerError.chatTemplate("No chat template was specified")
            // but that is not public so just fall back to text
            let prompt = input.prompt
                .asMessages()
                .compactMap { $0["content"] }
                .joined(separator: ". ")
            let promptTokens = tokenizer.encode(text: prompt)
            return LMInput(tokens: MLXArray(promptTokens))
        }
    }
}

@Observable
@MainActor
class LLMEvaluator {
    var running = false
    var output = ""
    var modelInfo = ""
    var stat = ""
    
    let basePrompt = """
    给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗：{title}
    """
    
    // Model configuration setup
//    let modelConfiguration = LLMModelFactory.shared.modelRegistry.configuration(id: "jerryzhao173985/poems")
    let modelConfiguration: ModelConfiguration = ModelConfiguration(
        directory: FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("huggingface/models")
            .appendingPathComponent("jerryzhao173985/poems"),
        defaultPrompt: "Tell me about the history of Spain."
    )

    /// parameters controlling the output
    let generateParameters = GenerateParameters(temperature: 0)
    let maxTokens = 240

    /// update the display every N tokens -- 4 looks like it updates continuously
    /// and is low overhead.  observed ~15% reduction in tokens/s when updating
    /// on every token
    let displayEveryNTokens = 4

    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    var loadState = LoadState.idle
    
    // App-specific local model directory
    let localModelDirectory: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("huggingface/models/jerryzhao173985/poems")
//    Hugging face hub HubApi: downloadBase
    let modelStorageDirectory: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("huggingface")

    // Function to check if model exists locally
    func modelExistsLocally() -> Bool {
        let modelPath = localModelDirectory.appendingPathComponent("model.safetensors")
        print("Checking if model exists locally at \(modelPath.path)")
        return FileManager.default.fileExists(atPath: modelPath.path)
    }
    
    
//    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    private func _load(
        configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContext {
        // load the generic config to unerstand which model and how to load the weights
        let configurationURL = localModelDirectory.appending(component: "config.json")
        let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: Data(contentsOf: configurationURL))
        
        let model = try LLMModelFactory.shared.typeRegistry.createModel(configuration: configurationURL, modelType: baseConfig.modelType)

        // apply the weights to the bare model
        try loadWeights(modelDirectory: localModelDirectory, model: model, quantization: baseConfig.quantization)

        let tokenizer = try await loadTokenizer(configuration: configuration, hub: HubApi(downloadBase: modelStorageDirectory))

        return .init(
            configuration: configuration, model: model,
            processor: LLMUserInputProcessor(tokenizer: tokenizer, configuration: configuration),
            tokenizer: tokenizer)
    }
    
    

    // Function to load model from local directory
    func loadModelFromLocal() async throws -> ModelContainer {
        self.modelInfo = "Loading model from local directory"
        
        // Load model container from the local directory
        let modelContainer = try await ModelContainer(context: _load(configuration: modelConfiguration, progressHandler: { _ in }))
//        LLMModelFactory.shared.loadContainer(hub:hubApi, configuration: modelConfiguration)
        return modelContainer
    }

    // Function to check the model remotely using Hub API and download if necessary
    func checkAndDownloadModel(hub: HubApi) async throws -> URL {
        let repo = Hub.Repo(id: "jerryzhao173985/poems")
        let modelFiles = ["*.safetensors", "*.json"]
        return try await hub.snapshot(from: repo, matching: modelFiles) { progress in
            Task { @MainActor in
                self.modelInfo = "Downloading \(self.modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
            }
        }
    }

    // Function to download the model if not available locally
    func downloadModel() async throws -> ModelContainer {
        let hub = HubApi()
        self.modelInfo = "Downloading model for \(modelConfiguration.name) from Hugging Face model hub"
        let modelDirectory = try await checkAndDownloadModel(hub: hub)
        print("Downloaded model to \(modelDirectory.path)")
        
        // After download, load the model from the local directory
        let modelContainer = try await LLMModelFactory.shared.loadContainer(configuration: modelConfiguration)
        
        // You can also perform additional model setup or saving here if needed
        
        return modelContainer
    }

    // Load model, check if it's already available or needs to be downloaded
    func load(hub: HubApi) async throws -> ModelContainer {
        switch loadState {
        case .idle:
          // limit the buffer cache
          MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
          let modelContainer = if modelExistsLocally() {
              try await loadModelFromLocal()
          } else {
              try await downloadModel()
          }

          let numParams = await modelContainer.perform { context in
              context.model.numParameters()
          }

          self.modelInfo =
              "Loaded \(modelConfiguration.id).  Weights: \(numParams / (1024*1024))M"
          loadState = .loaded(modelContainer)
          return modelContainer
        
        case .loaded(let modelContainer):
            return modelContainer
        }
          
    }

    // Function to generate model output based on user input
    func generate(prompt: String) async {
        guard !running else { return }

        running = true
        self.output = ""

        do {
            // Assuming `hub` is available
            let hub = HubApi() // Initialize the HubApi appropriately
            let modelContainer = try await load(hub: hub)

            // Create the prompt by interpolating the user input into the base prompt
            let fullPrompt = basePrompt.replacingOccurrences(of: "{title}", with: prompt)

            // Seed the random number generator
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            let result = try await modelContainer.perform { context in
                let input = try await context.processor.prepare(input: .init(prompt: fullPrompt))
                return try MLXLMCommon.generate(
                    input: input, parameters: generateParameters, context: context
                ) { tokens in
                    // Update the output
                    if tokens.count % displayEveryNTokens == 0 {
                        let text = context.tokenizer.decode(tokens: tokens)
                        Task { @MainActor in
                            self.output = text
                        }
                    }

                    if tokens.count >= maxTokens {
                        return .stop
                    } else {
                        return .more
                    }
                }
            }

            // Update the text if needed
            if result.output != self.output {
                self.output = result.output
            }
            self.stat = " Tokens/second: \(String(format: "%.3f", result.tokensPerSecond))"

        } catch {
            output = "Failed: \(error)"
        }

        running = false
    }
}

