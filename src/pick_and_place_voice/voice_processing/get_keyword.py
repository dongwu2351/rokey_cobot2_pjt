# ros2 service call /get_keyword std_srvs/srv/Trigger "{}"

import os
import re
from pathlib import Path

import rclpy
import pyaudio
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate  # d2 이거를 langchain_core로 바꿈
# from langchain.chains import LLMChain

from std_srvs.srv import Trigger
from voice_processing.MicController import MicController, MicConfig

from voice_processing.wakeup_word import WakeupWord
from voice_processing.stt import STT

############ Package Path & Environment Setting ############

#----------------------------------------------------------------
# current_dir = os.getcwd()
# package_path = get_package_share_directory("pick_and_place_voice")

# env_path = "/home/rokey/cobot_ws/src/cobot2_ws/pick_and_place_voice/resource/.env"
# load_dotenv(dotenv_path=env_path)
# is_load = load_dotenv(dotenv_path=os.path.join(f"{package_path}/resource/.env"))
# openai_api_key = os.getenv("OPENAI_API_KEY")
#-----------------------------------------------------------------

PACKAGE_NAME = "pick_and_place_voice"
PACKAGE_PATH = Path(get_package_share_directory(PACKAGE_NAME))
SOURCE_FILE = Path(__file__).resolve()
WORKSPACE_PATH = SOURCE_FILE.parents[2]
ENV_CANDIDATES = (
    Path(os.environ["VOICE_ENV_PATH"])
    if os.environ.get("VOICE_ENV_PATH")
    else Path("__VOICE_ENV_PATH_NOT_SET__"),
    PACKAGE_PATH / "resource" / ".env",
    SOURCE_FILE.parents[1] / "resource" / ".env",
    WORKSPACE_PATH / "corecode" / "VoiceProcessing" / ".env",
)
for env_path in ENV_CANDIDATES:
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path)
        break
openai_api_key = os.getenv("OPENAI_API_KEY")
TOOL_NAMES = ("drill", "hammer", "pliers", "screwdriver", "wrench")

############ AI Processor ############
# class AIProcessor:
#     def __init__(self):



############ GetKeyword Node ############
class GetKeyword(Node):
    def __init__(self):
        super().__init__("get_keyword_node")
        if not openai_api_key:
            checked = ", ".join(str(path) for path in ENV_CANDIDATES)
            raise RuntimeError(
                f"OPENAI_API_KEY is not set. Checked environment and: {checked}"
            )
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.0, openai_api_key=openai_api_key
        )

        prompt_content = """
            당신은 사용자의 문장에서 집어야 하는 도구 이름을 추출해야 합니다.

            <목표>
            - 문장에서 다음 리스트에 포함된 도구를 최대한 정확히 추출하세요.

            <도구 리스트>
            - drill, hammer, pliers, screwdriver, wrench

            <출력 형식>
            - 도구의 영문 이름만 공백으로 구분해서 출력하세요.
            - 설명, 문장, 괄호, 마크다운을 추가하지 마세요.
            - 도구가 없으면 NONE만 출력하세요.

            <특수 규칙>
            - 명확한 도구 명칭이 없지만 문맥상 유추 가능한 경우(예: "못 박는 것" → hammer)는 리스트 내 항목으로 최대한 추론해 반환하세요.

            <예시>
            - 입력: "망치를 집어줘"
            출력: hammer

            - 입력: "못 박는 것하고 드라이버를 가져와"
            출력: hammer screwdriver

            - 입력: "펜을 가져와"
            출력: NONE

            <사용자 입력>
            "{user_input}"                
        """

        self.prompt_template = PromptTemplate(
            input_variables=["user_input"], template=prompt_content
        )
        self.lang_chain = self.prompt_template | self.llm
        # self.lang_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        self.stt = STT(openai_api_key=openai_api_key)

        # 오디오 설정
        mic_config = MicConfig(
            chunk=12000,
            rate=48000,
            channels=1,
            record_seconds=5,
            fmt=pyaudio.paInt16,
            device_index=10,
            buffer_size=24000,
        )
        self.mic_controller = MicController(config=mic_config)
        # self.ai_processor = AIProcessor()

        self.get_logger().info("MicRecorderNode initialized.")
        self.get_logger().info("wait for client's request...")
        self.get_keyword_srv = self.create_service(
            Trigger, "get_keyword", self.get_keyword
        )
        self.wakeup_word = WakeupWord(mic_config.buffer_size)

    def extract_keyword(self, output_message):  # d2 이 함수 일부 수정함
        response = self.lang_chain.invoke({"user_input": output_message})
        result = response.content.lower()
        words = re.findall(r"[a-z]+", result)
        tools = []
        for word in words:
            if word in TOOL_NAMES and word not in tools:
                tools.append(word)
        print(f"LLM response: {response.content}")
        print(f"Tools: {tools}")
        return tools
    
    def get_keyword(self, request, response):  # 요청과 응답 객체를 받아야 함    # d2 이 함수 일부 수정함
        try:
            print("open stream")
            self.mic_controller.open_stream()
            self.wakeup_word.set_stream(self.mic_controller.stream)
            while not self.wakeup_word.is_wakeup():
                pass
        except OSError as error:
            self.get_logger().error(f"Failed to open/read audio stream: {error}")
            response.success = False
            response.message = "microphone unavailable"
            return response
        finally:
            # Wake-word PyAudio must release the input device before STT starts.
            self.mic_controller.close_stream()

        try:
            print("음성 녹음을 시작합니다. 5초 동안 말해주세요...")
            self.mic_controller.open_stream()
            self.mic_controller.record_audio()
            wav_data = self.mic_controller.get_wav_data()
            print("녹음 완료. STT에 전송 중...")
        except OSError as error:
            self.get_logger().error(f"Command recording failed: {error}")
            response.success = False
            response.message = "command recording failed"
            return response
        finally:
            self.mic_controller.close_stream()

        try:
            output_message = self.stt.speech2text(wav_data)
            keyword = self.extract_keyword(output_message)
        except Exception as error:
            self.get_logger().error(f"STT/keyword extraction failed: {error}")
            response.success = False
            response.message = str(error)
            return response

        self.get_logger().warn(f"Detected tools: {keyword}")
        response.success = bool(keyword)
        response.message = " ".join(keyword) if keyword else "no supported tool"
        return response


def main():  # d2 메인문 일부 수정
    rclpy.init()
    node = GetKeyword()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
