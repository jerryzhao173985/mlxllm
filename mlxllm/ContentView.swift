// Copyright © 2024 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import MarkdownUI
import Metal
import SwiftUI
import Tokenizers

struct ContentView: View {

    @State var prompt = ""
    @State var llm = LLMEvaluator()
    @Environment(DeviceStat.self) private var deviceStat

    enum displayStyle: String, CaseIterable, Identifiable {
        case plain, markdown
        var id: Self { self }
    }

    @State private var selectedDisplayStyle = displayStyle.markdown
    
    // Controls the ephemeral "Copied!" animation
    @State private var showCopyConfirmation = false

    var body: some View {
        VStack(alignment: .leading) {
            VStack {
                HStack {
                    Text(llm.modelInfo)
                        .textFieldStyle(.roundedBorder)

                    Spacer()

                    Text(llm.stat)
                }
                HStack {
                    Spacer()
                    if llm.running {
                        ProgressView()
                            .frame(maxHeight: 20)
                        Spacer()
                    }
                    Picker("", selection: $selectedDisplayStyle) {
                        ForEach(displayStyle.allCases, id: \.self) { option in
                            Text(option.rawValue.capitalized)
                                .tag(option)
                        }

                    }
                    .pickerStyle(.segmented)
                    #if os(visionOS)
                        .frame(maxWidth: 250)
                    #else
                        .frame(maxWidth: 150)
                    #endif
                }
            }
            // Use a ZStack so we can overlay the ephemeral "Copied!" message
            ZStack {
                // Scrollable area for the output
                // show the model output
                ScrollView(.vertical) {
                    ScrollViewReader { sp in
                        Group {
                            if selectedDisplayStyle == .plain {
                                Text(llm.output)
                                    .textSelection(.enabled)
                            } else {
                                Markdown(llm.output)
                                    .textSelection(.enabled)
                            }
                        }
                        .onChange(of: llm.output) { _, _ in
                            sp.scrollTo("bottom")
                        }

                        Spacer()
                            .frame(width: 1, height: 1)
                            .id("bottom")
                    }
                }
                // Detect tap on the entire scrollable area
                .onTapGesture {
                    guard !llm.output.isEmpty else { return }
                    copyToClipboard(prompt: prompt, response: llm.output)

                    // Trigger the "Copied!" animation
                    withAnimation {
                        showCopyConfirmation = true
                    }
                    // Hide the confirmation after 1.5 seconds
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                        withAnimation {
                            showCopyConfirmation = false
                        }
                    }
                }
                
                // Show ephemeral "Copied!" pop-up if tapped
                if showCopyConfirmation {
                    VStack {
                        Label("Copied!", systemImage: "checkmark.circle.fill")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(Color.green.opacity(0.9).cornerRadius(8))
                            .transition(.scale.combined(with: .opacity))
                    }
                    // Position it near the top (adjust to taste)
                    .padding(.top, 30)
                }
            }

            HStack {
                TextField("prompt", text: $prompt)
                    .onSubmit(generate)
                    .disabled(llm.running)
                    #if os(visionOS)
                        .textFieldStyle(.roundedBorder)
                    #endif
                Button("generate", action: generate)
                    .disabled(llm.running)
            }
        }
        #if os(visionOS)
            .padding(40)
        #else
            .padding()
        #endif
        .toolbar {
            ToolbarItem {
                Label(
                    "Memory Usage: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))",
                    systemImage: "info.circle.fill"
                )
                .labelStyle(.titleAndIcon)
                .padding(.horizontal)
                .help(
                    Text(
                        """
                        Active Memory: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))/\(GPU.memoryLimit.formatted(.byteCount(style: .memory)))
                        Cache Memory: \(deviceStat.gpuUsage.cacheMemory.formatted(.byteCount(style: .memory)))/\(GPU.cacheLimit.formatted(.byteCount(style: .memory)))
                        Peak Memory: \(deviceStat.gpuUsage.peakMemory.formatted(.byteCount(style: .memory)))
                        """
                    )
                )
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task {
                        copyToClipboard(prompt: prompt, response: llm.output)
//                        copyToClipboard(llm.output)
                    }
                } label: {
                    Label("Copy Output", systemImage: "doc.on.doc.fill")
                }
                .disabled(llm.output.isEmpty)
//                .disabled(llm.output == "")
                .labelStyle(.titleAndIcon)
            }

        }
        .task {
            self.prompt = llm.modelConfiguration.defaultPrompt

            // pre-load the weights on launch to speed up the first generation
            _ = try? await llm.load()
        }
    }

    private func generate() {
        Task {
            await llm.generate(prompt: prompt)
        }
    }
//    private func copyToClipboard(_ string: String) {
//        #if os(macOS)
//            NSPasteboard.general.clearContents()
//            NSPasteboard.general.setString(string, forType: .string)
//        #else
//            UIPasteboard.general.string = string
//        #endif
//    }
    
    private func copyToClipboard(prompt: String, response: String) {
        // Format text as <input>\n<response>
//        let formattedText = "\(prompt)\n\(response)"
        let formattedText = "\(response)"

        #if os(macOS)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(formattedText, forType: .string)
        #else
            UIPasteboard.general.string = formattedText
        #endif
    }
}
