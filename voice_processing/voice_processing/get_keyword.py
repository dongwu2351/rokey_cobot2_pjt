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
from .MicController import MicController, MicConfig
from .wakeup_word import WakeupWord
from .stt import STT

############ Package Path & Environment Setting ############

#----------------------------------------------------------------
# current_dir = os.getcwd()
# package_path = get_package_share_directory("pick_and_place_voice")

# env_path = "/home/rokey/cobot_ws/src/cobot2_ws/pick_and_place_voice/resource/.env"
# load_dotenv(dotenv_path=env_path)
# is_load = load_dotenv(dotenv_path=os.path.join(f"{package_path}/resource/.env"))
# openai_api_key = os.getenv("OPENAI_API_KEY")
#-----------------------------------------------------------------

PACKAGE_NAME = "voice_processing"
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
TARGET_SETS = {
    "tools": ("drill", "hammer", "pliers", "screwdriver", "wrench"),
    "fruits": ("apple", "banana", "kiwi", "orange", "pear"),
}

############ AI Processor ############
# class AIProcessor:
#     def __init__(self):



############ GetKeyword Node ############
class GetKeyword(Node):
    def __init__(self):
        super().__init__("get_keyword_node")
        self.declare_parameter("target_set", "tools")
        self.target_set = (
            self.get_parameter("target_set")
            .get_parameter_value()
            .string_value
            .lower()
        )
        if self.target_set not in TARGET_SETS:
            choices = ", ".join(TARGET_SETS)
            raise ValueError(
                f"Unknown target_set '{self.target_set}'. Choose: {choices}"
            )
        self.target_names = TARGET_SETS[self.target_set]
        if not openai_api_key:
            checked = ", ".join(str(path) for path in ENV_CANDIDATES)
            raise RuntimeError(
                f"OPENAI_API_KEY is not set. Checked environment and: {checked}"
            )
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.0, openai_api_key=openai_api_key
        )

        prompt_content = """
            당신은 사용자의 문장에서 로봇이 집어야 하는 대상 이름을 추출해야 합니다.

            <목표>
            - 문장에서 다음 리스트에 포함된 대상을 최대한 정확히 추출하세요.

            <현재 대상 종류>
            - {target_set}

            <대상 리스트>
            - {target_names}

            <출력 형식>
            - 대상의 영문 이름만 공백으로 구분해서 출력하세요.
            - 설명, 문장, 괄호, 마크다운을 추가하지 마세요.
            - 대상이 없으면 NONE만 출력하세요.

            <특수 규칙>
            - tools 모드에서 "못 박는 것"은 hammer로 해석할 수 있습니다.
            - fruits 모드에서 사과=apple, 바나나=banana, 키위=kiwi,
              오렌지/귤=orange, 과일 배=pear로 해석하세요.
            - fruits 모드의 "배"는 과일을 집는 문맥일 때만 pear로 반환하세요.

            <예시>
            - 입력: "망치를 집어줘"
            출력: hammer

            - 입력: "못 박는 것하고 드라이버를 가져와"
            출력: hammer screwdriver

            - 입력: "펜을 가져와"
            출력: NONE

            - 입력: "사과를 집어줘"
            출력: apple

            - 입력: "배를 가져다줘"
            출력: pear

            <사용자 입력>
            "{user_input}"                
        """.replace("{target_set}", self.target_set).replace(
            "{target_names}", ", ".join(self.target_names)
        )

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
        targets = []
        for word in words:
            if word in self.target_names and word not in targets:
                targets.append(word)
        print(f"LLM response: {response.content}")
        print(f"Targets: {targets}")
        return targets
    
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

        self.get_logger().warn(
            f"Detected {self.target_set} targets: {keyword}"
        )
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
