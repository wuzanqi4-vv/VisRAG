import sys
sys.path.append("/home/wuzanqi/VisRAG/visrag_scripts/baseline_eval")
import config
from generate_qwen import QwenGenerator

generator = QwenGenerator(
    api_key=config.QWEN_API_KEY,
    base_url=config.QWEN_BASE_URL,
    model_name=config.QWEN_MODEL_NAME,
)

print("Testing API connectivity with model:", config.QWEN_MODEL_NAME)
res = generator.generate_answer(
    question="请回答：1+1等于几？只需回答数字即可。",
    image_paths=[],  # 空图片列表测试纯文本基础能力
)
print("API Response Result:", res)
