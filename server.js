// server.js - OpenAI to NVIDIA NIM API Proxy (Speed Optimized)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = false;

// 🔥 THINKING MODE TOGGLE
const ENABLE_THINKING_MODE = false;

// ⚡ SPEED OPTIMIZATION SETTINGS
const REQUEST_TIMEOUT = 300000; // 5 minutes
const MAX_RETRIES = 2;

// 🚀 VERIFIED FREE ENDPOINT MODELS - Exact names from NVIDIA NIM
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta/llama-3.3-70b-instruct',
  'gpt-4': 'meta/llama-3.3-70b-instruct',
  'gpt-4-turbo': 'meta/llama-3.1-405b-instruct',
  'gpt-4o': 'z-ai/glm4.7', // EXACT from NVIDIA: z-ai/glm4.7 (no hyphen)
  'claude-3-opus': 'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet': 'meta/llama-3.3-70b-instruct',
  'gemini-pro': 'meta/llama-3.1-70b-instruct'
};

// 🔄 FALLBACK CHAIN - All FREE ENDPOINT models only
const FALLBACK_CHAIN = [
  'z-ai/glm4.7',                     // #1 Best for roleplay/instruction following
  'meta/llama-3.3-70b-instruct',     // #2 Fast, great writing
  'meta/llama-3.1-70b-instruct',     // #3 Reliable
  'meta/llama-3.1-8b-instruct'       // #4 Fast fallback
];

// Helper function with retry + fallback logic
async function makeNIMRequest(nimRequest, stream, retryCount = 0) {
  try {
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: REQUEST_TIMEOUT,
      maxContentLength: 50 * 1024 * 1024,
      maxBodyLength: 50 * 1024 * 1024
    });
    return response;
  } catch (error) {
    // Retry logic for timeouts and server errors
    const isTimeout = error.code === 'ECONNABORTED' || error.message.includes('timeout');
    const isServerError = error.response?.status >= 500;
    
    if ((isTimeout || isServerError) && retryCount < MAX_RETRIES) {
      console.log(`⚠️  Request failed (${error.message}), trying fallback model (${retryCount + 1}/${MAX_RETRIES})`);
      
      // Switch to next fallback model
      if (FALLBACK_CHAIN[retryCount]) {
        const fallbackModel = FALLBACK_CHAIN[retryCount];
        console.log(`🔄 Switching to: ${fallbackModel}`);
        nimRequest.model = fallbackModel;
        return makeNIMRequest(nimRequest, stream, retryCount + 1);
      }
    }
    throw error;
  }
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy (Speed Optimized)', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    timeout_ms: REQUEST_TIMEOUT,
    max_retries: MAX_RETRIES
  });
});

// List models endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  const startTime = Date.now();
  console.log(`\n[${new Date().toISOString()}] 📨 Request started - Model: ${req.body.model}`);
  
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Get mapped model
    let nimModel = MODEL_MAPPING[model] || FALLBACK_CHAIN[0];
    
    // Transform request
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.9,
      max_tokens: max_tokens || 2048,
      // GLM-4.7 specific thinking parameters
      extra_body: (ENABLE_THINKING_MODE && nimModel === 'z-ai/glm4.7') 
        ? { chat_template_kwargs: { enable_thinking: true, clear_thinking: false } } 
        : undefined,
      stream: stream !== false
    };
    
    console.log(`🎯 Using NVIDIA model: ${nimModel}`);
    
    // Make request with retry logic
    const response = await makeNIMRequest(nimRequest, stream);
    
    if (stream) {
      // Streaming response
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      let firstChunkSent = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write('data: [DONE]\n\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              
              // Handle GLM-4.7 specific response format
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                // Always delete reasoning_content to avoid confusing clients
                delete data.choices[0].delta.reasoning_content;
                
                if (SHOW_REASONING) {
                  // Combine reasoning and content
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '\n</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                  }
                } else {
                  // Only send actual content, skip reasoning
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else if (!reasoning) {
                    // Keep empty content for role/finish_reason messages
                    data.choices[0].delta.content = data.choices[0].delta.content || '';
                  } else {
                    // Skip chunks that only have reasoning
                    return;
                  }
                }
              }
              
              // Send the chunk
              res.write(`data: ${JSON.stringify(data)}\n\n`);
              firstChunkSent = true;
            } catch (e) {
              // Invalid JSON, pass through as-is
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
        console.log(`✅ Stream completed successfully in ${elapsed}s`);
        if (!firstChunkSent) {
          console.warn('⚠️  Warning: Stream ended but no chunks were sent to client');
        }
        res.write('data: [DONE]\n\n');
        res.end();
      });
      
      response.data.on('error', (err) => {
        console.error('❌ Stream error:', err);
        res.write(`data: ${JSON.stringify({"error": "Stream interrupted: " + err.message})}\n\n`);
        res.end();
      });
      
      req.on('close', () => {
        console.log('🔌 Client disconnected');
        response.data.destroy();
      });
    } else {
      // Non-streaming response
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
      console.log(`✅ Response completed in ${elapsed}s`);
      res.json(openaiResponse);
    }
    
  } catch (error) {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
    console.error(`❌ Proxy error after ${elapsed}s:`, error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`\n🚀 OpenAI to NVIDIA NIM Proxy (Speed Optimized)`);
  console.log(`📡 Running on port ${PORT}`);
  console.log(`🏥 Health check: http://localhost:${PORT}/health`);
  console.log(`⏱️  Timeout: ${REQUEST_TIMEOUT/1000}s`);
  console.log(`🔄 Max retries: ${MAX_RETRIES}`);
  console.log(`💭 Reasoning: ${SHOW_REASONING ? 'ON' : 'OFF'}`);
  console.log(`🤔 Thinking mode: ${ENABLE_THINKING_MODE ? 'ON' : 'OFF'}`);
  console.log(`\n🎯 Primary models:`);
  Object.entries(MODEL_MAPPING).forEach(([key, value]) => {
    console.log(`   ${key} → ${value}`);
  });
  console.log(`\n`);
});