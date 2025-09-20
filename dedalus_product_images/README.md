# Dedalus Product Images Workflow

This is a modified version of the original product images workflow that uses Dedalus agents for both prompt generation and image generation. Uses Dedalus agents throughout the process instead of direct LangGraph functions.

## 🏗️ Architecture Changes

### **Original vs Dedalus Version**

| Component | Original | Dedalus Version |
|-----------|----------|-----------------|
| **Prompt Generation** | LangGraph Node + OpenAI | **Dedalus Agent + OpenAI** |
| **Image Generation** | FluxAdapter | **Dedalus Agent + Custom MCP** |
| **File Management** | LangGraph Node | LangGraph Node |
| **Workflow Orchestration** | LangGraph | LangGraph |

### **Key Changes**

1. **`prompt_and_generate` → `dedalus_prompt_and_generate`**
   - Now uses Dedalus agents for both prompt generation and image generation
   - Two-step Dedalus process: prompt generation → image generation
   - Uses custom MCP tools for image generation
   - Maintains same input/output interface

2. **Dual Dedalus Agent Process**
   - **Step 1**: Dedalus agent generates diffusion prompts using OpenAI
   - **Step 2**: Dedalus agent generates images using custom MCP tools
   - Both steps use the same Dedalus client and runner

3. **MCP Integration**
   - Uses custom image generation MCP tools
   - Response parsing functions for MCP tool outputs
   - Easy to customize based on your MCP tool's response format

## 🔧 Setup Required

### **1. Environment Variables**

Ensure you have:
- `OPENAI_API_KEY` - for Dedalus agent prompt generation
- Dedalus Labs configuration
- Your custom image generation MCP tool properly configured

### **2. Dependencies**

Install the required dependencies:
- `langgraph`
- `langchain-openai`
- `dedalus-labs`
- `pydantic`
- `requests` (for downloading images from URLs)

### **3. MCP Server Configuration**

The workflow is configured to use the Flux MCP server:

```python
# In product_images_graph.py, line ~309
mcp_servers=["yihu/flux-mcp"]  # Flux image generation MCP server
```

### **4. Important: Dedalus Limitations**

⚠️ **Dedalus Labs does NOT support structured output** like LangChain's `.with_structured_output()`. The implementation handles this by:

1. **Prompt Engineering**: Using specific prompts to get consistent text responses
2. **Response Parsing**: Parsing unstructured text responses from Dedalus agents
3. **Multiple Format Support**: Handling base64, URLs, and file paths in responses

### **5. Flux MCP Response Format**

The `yihu/flux-mcp` server should return image data in one of these formats:
- **Base64**: `data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...`
- **URL**: `https://example.com/generated-image.png`
- **File Path**: `/path/to/generated/image.png`

The `parse_and_save_image_response` function automatically detects and handles these formats.

## 🚀 Usage

### **Run the Workflow**

```bash
python test_graph.py
```

### **Expected Output**

The workflow will:
1. **Parse product description** and create variation dimensions
2. **Generate 9 parallel requests** (3×3 dimension combinations)
3. **Use Dedalus agents** for both prompt generation and image generation
4. **Save results** to organized directories

## 📁 File Structure

```
dedalus_product_images/
├── __init__.py
├── product_images_graph.py    # Main workflow with Dedalus integration
├── prompts.py                 # Prompt templates (unchanged)
├── test_graph.py             # Test script
├── README.md                 # This documentation
└── gen_images/               # Generated outputs (created at runtime)
    └── run_YYYYMMDD_HHMMSS_xxxxxx/
        ├── images/           # Generated images
        ├── prompts/          # Generated prompts
        └── config.jsonl      # Run configuration
```

## 🔄 Workflow Flow

```
START → init_run_output → propose_dimensions → save_run_config → start_variation_fanout
                                                                    ↓
                                                              split_to_variations
                                                                    ↓
                                                              [Parallel Execution]
                                                                    ↓
                                                              dedalus_prompt_and_generate (×9)
                                                                    ↓
                                                              finalize → END
```

## 🎯 Key Benefits

1. **MCP Integration**: Leverages Dedalus Labs' MCP ecosystem
2. **Custom Image Tools**: Use your specialized image generation MCP
3. **Maintained Workflow**: Same LangGraph orchestration and file management
4. **Parallel Processing**: Still generates 9 images simultaneously
5. **Extensible**: Easy to add more MCP tools or modify the workflow

## 🔮 Future Enhancements

- **Multiple MCP Tools**: Add different image generation services
- **Quality Assessment**: Use MCP tools to evaluate generated images
- **Style Transfer**: Use specialized MCP tools for style variations
- **Batch Processing**: Optimize for larger image generation runs

## 🐛 Troubleshooting

### **Common Issues**

1. **MCP Server Not Found**
   - Ensure your MCP server is properly configured in Dedalus Labs
   - Check the MCP server identifier matches your configuration

2. **Response Parsing Errors**
   - Customize `extract_prompt_from_response` and `save_generated_image`
   - Check your MCP tool's response format

3. **Image Generation Failures**
   - Verify your MCP tool is working correctly
   - Check Dedalus Labs logs for detailed error messages

### **Debug Mode**

Add debug logging to see Dedalus agent responses:

```python
print(f"🔍 Dedalus Response: {response.final_output}")
```

## 📚 Next Steps

1. **Configure your custom image generation MCP**
2. **Test the workflow with a simple product description**
3. **Customize response parsing based on your MCP tool**
4. **Optimize the workflow for your specific use case**
5. **Add additional MCP tools for enhanced functionality**

The workflow is ready to use once you configure your custom MCP tool! 🎨✨
