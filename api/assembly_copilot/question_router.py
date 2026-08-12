from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AssemblyQuestion:
    intent: str
    text: str
    claimed_step_id: str | None = None


class AssemblyQuestionRouter:
    STEP = re.compile(
        r"(?:step|스텝)\s*([0-9일이삼사오육칠팔구십]+)(?:\s*단계)?|"
        r"([0-9일이삼사오육칠팔구십]+)\s*단계", re.I)
    CHECK = re.compile(r"(잘|맞|제대로|정상|괜찮|확인|검사|문제|잘못)")
    CURRENT = re.compile(
        r"(현재.*(?:단계|스텝|상태|작업)|"
        r"몇\s*(?:단계|스텝)|어느\s*(?:단계|스텝)|"
        r"어디까지.*(?:했|됐|왔)|무슨\s*작업|"
        r"뭐\s*(?:하고|하는)\s*(?:있|중)|작업\s*상태)", re.I,
    )
    NEXT = re.compile(r"(다음|뭘\s*해야|무엇을\s*해야)")
    ESTIMATE_STAGE = re.compile(
        r"(?:어디까지|어떤\s*단계까지|어느\s*단계까지).*?"
        r"(?:완료|진행|한\s*것).*?(?:같|판단|확인)", re.I)
    DESCRIBED_PROGRESS = re.compile(
        r"(?:걸었|걸고|끼웠|끼우고|연결(?:까지)?\s*(?:했|하고|했는데)|조였|조이고|체결했|체결하고|"
        r"고정했|고정하고|장착했|장착하고|붙였|넣었|완료했|다\s*했).*?"
        r"(?:다음|뭘\s*해야|무엇을\s*해야)", re.I)
    WORK_REPORT = re.compile(
        r"(?:타이밍\s*벨트|모터\s*조립체|너트|볼트|풀리|브래킷|다공판|레일|프로파일)"
        r".*?(?:걸었|끼웠|연결했|고정했|조였|체결했|장착했|붙였)", re.I)
    SHOW = re.compile(r"(사진|이미지|영상|예시|정상\s*모양|보여|띄워)")
    TARGET_INFO = re.compile(
        r"(뭐(?:야|지|인지)|무엇|어떻게|방법|과정|설명|알려|안내|해야\s*해|하는\s*거)", re.I)
    SELECT = re.compile(
        r"(?:단계|스텝)(?:로|부터).*?(?:가|넘어|진행|변경|바꿔|시작|할게)|"
        r"(?:가|넘어|진행|변경|바꿔|시작).*?(?:단계|스텝)", re.I)
    RETURN = re.compile(
        r"(?:단계|스텝)(?:로|으로)?.*?(?:돌아(?:가|갈)|되돌려|다시\s*하)|"
        r"(?:돌아(?:가|갈)|되돌려|다시\s*하).*?(?:단계|스텝)", re.I)
    STEP_MAPPING_CHALLENGE = re.compile(
        r"(?:인\s*걸로|라고)\s*(?:알|기억).*?(?:왜|맞)|"
        r"왜.*?[0-9일이삼사오육칠팔구십]+\s*단계.*?(?:설명|안내|말)", re.I)
    COMPLETE = re.compile(
        r"(다\s*했|완료(?:했|됐|야)?|끝났|끝냈|마쳤|조립했|작업했|"
        r"다음\s*단계로\s*(?:넘어|진행))"
    )
    QUESTION = re.compile(
        r"(?:나요|까요|습니까|인가요|인지|맞아|맞나요|"
        r"했어\?|됐어\?|확인해|것\s*같|거\s*같|같아\?)", re.I)

    def route(self, text: str) -> AssemblyQuestion:
        match = self.STEP.search(text)
        number = next((value for value in match.groups() if value), None) if match else None
        parsed = _step_number(number) if number else None
        claimed = f"step_{parsed:02d}" if parsed else None
        # Completion words in an interrogative are inspection requests, not a
        # command to mutate assembly state.
        if claimed and self.STEP_MAPPING_CHALLENGE.search(text):
            intent = "CHALLENGE_STEP_MAPPING"
        elif claimed and self.RETURN.search(text):
            intent = "RETURN_TO_STEP"
        elif self.ESTIMATE_STAGE.search(text):
            intent = "IDENTIFY_STEP"
        elif claimed and self.SELECT.search(text) and not self.QUESTION.search(text):
            intent = "SELECT_STEP"
        elif self.COMPLETE.search(text) and not self.QUESTION.search(text):
            intent = "COMPLETE_STEP"
        elif self.COMPLETE.search(text) and self.QUESTION.search(text):
            intent = "CHECK_CLAIMED_STEP" if claimed else "CHECK_PROGRESS"
        elif (self.CURRENT.search(text) or self.DESCRIBED_PROGRESS.search(text)
              or self.WORK_REPORT.search(text)):
            intent = "IDENTIFY_STEP"
        elif claimed and self.SHOW.search(text) and self.TARGET_INFO.search(text):
            intent = "EXPLAIN_AND_SHOW_TARGET"
        elif claimed and self.SHOW.search(text):
            intent = "SHOW_TARGET_REFERENCE"
        elif claimed and self.TARGET_INFO.search(text):
            intent = "EXPLAIN_TARGET_STEP"
        elif self.CHECK.search(text) or claimed:
            intent = "CHECK_CLAIMED_STEP" if claimed else "CHECK_PROGRESS"
        elif self.NEXT.search(text):
            intent = "NEXT_STEP"
        elif self.SHOW.search(text):
            intent = "SHOW_REFERENCE"
        else:
            intent = "EXPLAIN_STEP"
        return AssemblyQuestion(intent, text.strip(), claimed)


def _step_number(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    digits = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
              "육": 6, "칠": 7, "팔": 8, "구": 9}
    if "십" in token:
        tens, ones = token.split("십", 1)
        return digits.get(tens, 1) * 10 + digits.get(ones, 0)
    return digits.get(token)
