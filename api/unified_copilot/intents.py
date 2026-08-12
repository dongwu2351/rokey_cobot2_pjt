from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class UnifiedIntent:
    domain: str
    intent: str
    text: str


class UnifiedIntentRouter:
    """Cheap, deterministic gates before open-ended conversation LLM calls."""

    SHOW_DISPLAY = re.compile(r"((?:화면|창|카메라|ar).*?(?:보여|열어|켜|띄워)|(?:보여|열어|켜|띄워).*?(?:화면|창|카메라|ar))", re.I)
    SHOW_LIVE_DISPLAY = re.compile(
        r"(?:(?:현재|실제|작업\s*중인).*?작업|"
        r"작업대|카메라|RealSense|리얼센스).*?"
        r"(?:화면|이미지|영상).*?(?:보여|띄워|띄우|열어|표시)|"
        r"참고\s*(?:사진|이미지)\s*말고.*?"
        r"(?:현재|실제|작업\s*중|작업대|카메라).*?"
        r"(?:화면|이미지|영상)|"
        r"(?:내가|제가).*?작업\s*중인.*?(?:화면|이미지).*?"
        r"(?:보여|띄워|띄우|열어|표시)|"
        r"왜\s*안\s*(?:띄워|띄우|보여|열어)(?:주는)?\s*거", re.I)
    SHOW_REFERENCE = re.compile(
        r"((?:사진|이미지|참고\s*(?:사진|이미지|자료)|정상\s*(?:사진|이미지)).*?(?:보여|띄워|열어)|"
        r"(?:보여|띄워|열어).*?(?:사진|이미지|참고\s*자료))", re.I)
    DISPLAY_FEEDBACK = re.compile(
        r"(?:왜|뭐\s*하러).*?(?:사진|이미지|참고\s*자료).*?"
        r"(?:보여|띄워|표시)|"
        r"(?:사진|이미지|참고\s*자료).*?(?:요청한\s*적|요청한\s*기억|"
        r"요청하지|요청한\s*게).*?(?:없|아니)|"
        r"(?:사진|이미지|참고\s*자료).*?(?:보여|띄워).*?"
        r"(?:냐고|거야|거냐)", re.I)
    STATE_FEEDBACK = re.compile(
        r"(?:어디까지|완료\s*여부).*?(?:모르|확인\s*안).*?"
        r"왜.*?(?:완료|다음|넘어|변경|기록)|"
        r"왜.*?(?:완료|단계).*?(?:처리|기록|넘어|변경|설명|안내)", re.I)
    VISION_FEEDBACK = re.compile(
        r"유사한\s*단계.*?(?:알려|요청).*?(?:없|안)|"
        r"왜.*?유사한\s*단계.*?(?:설명|안내|말)", re.I)
    STEP_MAPPING_CHALLENGE = re.compile(
        r"[0-9일이삼사오육칠팔구십]+\s*단계.*?"
        r"(?:인\s*걸로|라고)\s*(?:알|기억).*?왜.*?"
        r"[0-9일이삼사오육칠팔구십]+\s*단계", re.I)
    RETURN_TO_STEP = re.compile(
        r"[0-9일이삼사오육칠팔구십]+\s*단계.*?"
        r"(?:덜\s*된|미완료|못\s*한|안\s*한).*?"
        r"(?:돌아(?:가|갈)|되돌려|다시)|"
        r"[0-9일이삼사오육칠팔구십]+\s*단계(?:로|으로).*?"
        r"(?:돌아(?:가|갈)|되돌려|다시)", re.I)
    CAMERA_FEEDBACK = re.compile(
        r"(?:리얼센스|realsense|카메라).*?(?:보이|켜져|연결).*?"
        r"왜.*?인식.*?(?:못|안)|"
        r"(?:리얼센스|realsense|카메라).*?(?:화면|이미지|프레임).*?"
        r"(?:없|안\s*들어|안\s*보|전달\s*안|못\s*보|"
        r"인식\s*(?:안|못)|왜\s*인식)|"
        r"(?:리얼센스|realsense|카메라).*?(?:프레임|이미지).*?"
        r"(?:가져|받아).*?(?:분석|인식).*?(?:해야|하는)|"
        r"(?:화면|이미지|프레임).*?(?:없|전달되지).*?"
        r"(?:어떻게|왜|카메라|리얼센스)", re.I)
    START_WORK = re.compile(
        r"((?:작업|조립)\s*(?:을\s*)?(?:시작|진행)(?:하자|해|해줘|할게)?|"
        r"(?:시작|진행)(?:하자|해|해줘).*?(?:작업|조립)|"
        r"(?:지금부터|이제)\s*(?:작업을?\s*)?시작해\s*볼까|"
        r"(?:작업|조립)\s*(?:을\s*)?(?:재개|계속|이어)(?:해|하자|할게|해줘)?|"
        r"이전\s*(?:작업|조립).*?(?:이어|계속)|"
        r"(?:step|스텝|단계)?\s*\d+\s*(?:단계|스텝)?\s*(?:조립|작업|시작|진행))", re.I)
    HIDE_DISPLAY = re.compile(r"((?:화면|창|카메라|ar).*?(?:닫아|닫아줘|숨겨|꺼|그만)|(?:닫아|숨겨|꺼).*?(?:화면|창|카메라|ar))", re.I)
    SAVE_PROGRESS_AND_HIDE = re.compile(
        r"[0-9일이삼사오육칠팔구십]+\s*단계.*?(?:기억|저장).*?(?:화면|창|ar).*?(?:닫|숨기|꺼)|"
        r"(?:기억|저장).*?[0-9일이삼사오육칠팔구십]+\s*단계.*?(?:화면|창|ar).*?(?:닫|숨기|꺼)", re.I)
    SAVE_PROGRESS_AND_STOP = re.compile(
        r"[0-9일이삼사오육칠팔구십]+\s*단계.*?(?:했|완료|마쳤|하고).*?"
        r"(?:종료|끝내|그만)|[0-9일이삼사오육칠팔구십]+\s*단계.*?"
        r"(?:기억|저장).*?(?:종료|끝내|그만)", re.I)
    STOP_COPILOT = re.compile(
        r"(?:코파일럿|코파일러|프로그램|앱|애플리케이션|전체)\s*(?:을\s*)?"
        r"(?:종료|끝내|꺼|닫아)(?:\s*(?:줘|주세요))?", re.I)
    PAUSE_WORK = re.compile(
        r"(?:작업|조립)\s*(?:을\s*)?(?:일시\s*정지|중단|종료|그만|멈춰)|"
        r"(?:작업|조립)(?:은|는)?\s*여기까지", re.I)
    AMBIGUOUS_STOP = re.compile(
        r"^(?:종료|끝내|꺼|그만|멈춰|quit|exit|stop)"
        r"(?:\s*(?:해(?:\s*줘|주세요)?|시켜\s*줘|줄래))?[.!?]?$", re.I)
    PROGRESS_UPDATE = re.compile(
        r"[0-9일이삼사오육칠팔구십]+\s*단계\s*(?:까지)?.*?(?:완료|했|마쳤|확인).*?"
        r"(?:(?:상태|단계|진행|작업).*?)?(?:업데이트|변경|반영|기록|저장)|"
        r"(?:상태|단계|진행|작업).*?(?:업데이트|변경|반영|기록|저장).*?"
        r"[0-9일이삼사오육칠팔구십]+\s*단계\s*(?:까지)?.*?(?:완료|했|마쳤|확인)", re.I)
    ACTIVE_STEP_UPDATE = re.compile(
        r"(?:현재|지금).*?(?:단계|작업\s*위치).*?"
        r"[0-9일이삼사오육칠팔구십]+\s*단계(?:로|부터)?.*?"
        r"(?:업데이트|변경|반영|설정|맞춰)|"
        r"[0-9일이삼사오육칠팔구십]+\s*단계(?:로|부터).*?"
        r"(?:현재|지금).*?(?:상태|단계|작업\s*위치).*?"
        r"(?:업데이트|변경|반영|설정|맞춰)|"
        r"(?:화면|표시).*?(?:스텝|단계).*?"
        r"[0-9일이삼사오육칠팔구십]+(?:\s*단계)?(?:로|으로)?.*?"
        r"(?:수정|변경|바꿔|맞춰)", re.I)
    CANCEL = re.compile(
        r"^(?:아니|아니요)(?:[,\s]+(?:그렇지\s*않아|안\s*보여|아닌\s*것\s*같아))?|"
        r"(?:취소|하지\s*마|그대로\s*둬)[.!?]?$", re.I)
    # Spoken answers arrive punctuated ("응, 가져와") - the separator after the
    # yes-word must swallow the comma or the whole answer misses and the
    # request is re-asked forever.
    CONFIRM = re.compile(
        r"^(?:(?:네|응|어|엉|음|그래|좋아|오케이|ok)"
        r"(?:[,\s]+(?:그렇게|그거|그걸))?[,\s]*"
        r"(?:해(?:\s*줘)?|진행해(?:\s*줘)?|바꿔(?:\s*줘)?|수정해(?:\s*줘)?|시작해|"
        r"가져(?:와|다\s*줘)(?:\s*줘)?|올려\s*줘)?|"
        r"(?:네|응|예)\s*(?:맞아(?:요)?|보여(?:요)?|그래요|그렇습니다)|"
        r"확인|진행|교체해|다시\s*만들어|그렇게\s*해(?:\s*줘)?|"
        r"바꿔(?:\s*줘)?|넘어가(?:\s*줘)?|가져와(?:\s*줘)?|가져다\s*줘|"
        r"(?:바로\s*)?넘어가도\s*(?:돼|됩니다|괜찮아))[.!?]?$", re.I)
    # Spoken agreement is open-ended - "어 가져와", "응 부탁해", "그래 해줘",
    # "당연하지", "ㅇㅇ". Enumerating every phrasing loses, so while a question
    # is pending the rule is shape-based: a SHORT answer that opens with an
    # agreement token (or is a bare "please do it" verb) means yes, unless it
    # carries a refusal/postponement word. Wrongly hearing yes only starts a
    # motion the user can stop by voice; wrongly hearing no strands them
    # repeating themselves, which is what actually happened.
    CONFIRM_HEAD = re.compile(
        r"^(?:네+|넵|넹|예+|응+|어+|엉+|음+|ㅇㅇ+|오케이|오키|ok(?:ay)?|"
        r"그래+|그럼+|그렇지|그러(?:자|지)|당연(?:하지|히)?|물론|좋아+|좋지|"
        r"맞아+|맞습니다|굿|고고|해도\s*(?:돼|좋아)|부탁)", re.I)
    #: Bare polite imperatives that answer "진행할까요?" without a yes-token.
    CONFIRM_VERB = re.compile(
        r"^(?:부탁(?:해|할게|드려|드릴게)?(?:요)?|해\s*줘|해줘|진행(?:해|하자)?|"
        r"시작(?:해|하자)?|가져(?:와|다)|가져와|올려\s*줘|고고|가자|시켜)", re.I)
    #: Refusal/postponement inside an otherwise affirmative-looking answer.
    CONFIRM_NEGATIVE = re.compile(
        r"아니|안\s*(?:돼|해|가져)|하지\s*마|취소|그만|잠깐|잠시만|나중에|말고|"
        r"싫|됐어|필요\s*없", re.I)

    MANUAL = re.compile(r"(pdf|피디에프|조립서|매뉴얼).*(찾|확인|분석|생성|만들|갱신|교체|다시)|다운로드.*(?:조립서|매뉴얼|pdf)", re.I)
    SEARCH = re.compile(
        r"(인터넷|웹)\s*(?:에서\s*)?(?:검색|찾)|(?:검색|찾아봐|찾아줘).*?(?:인터넷|웹)|"
        r"(?:오늘|내일|현재|지금).*?(?:날씨|기온|미세먼지)|"
        r"(?:날씨|기온|미세먼지|최신\s*뉴스|현재\s*가격|주소)\s*(?:알려|검색|찾)|"
        r"(?:내\s*위치|근처|주변).*?(?:맛집|식당|카페|병원|약국|가볼\s*곳).*?(?:찾|알려|추천)?|"
        r"(?:맛집|식당|카페).*?(?:찾아|검색|추천)", re.I)
    DESCRIBE = re.compile(r"(이\s*(?:물체|부품)|들고\s*있는|카메라.*(?:뭐|무엇)|이게\s*뭐)", re.I)
    WORK_REPORT = re.compile(
        r"(?:타이밍\s*벨트|모터\s*조립체|너트|볼트|풀리|브래킷|다공판|레일|프로파일)"
        r".*?(?:걸었|끼웠|연결했|고정했|조였|체결했|장착했|붙였|완료했)", re.I)
    ASSEMBLY = re.compile(
        r"(현재.*(?:단계|스텝|조립|작업\s*상태)|몇\s*(?:단계|스텝)|"
        r"(?:어떤|어느|어디)\s*단계까지.*?(?:완료|진행).*?(?:같|판단|확인)|"
        r"어디까지.*?(?:완료|진행).*?(?:같|판단|확인)|"
        r"어디까지.*(?:했|됐|왔)|잘\s*(?:하고|조립)|맞(?:아|나요)|"
        r"조립.*(?:확인|검사|문제)|다음.*(?:단계|작업|뭘|무엇)|"
        r"단계.*(?:완료|끝)|다\s*했어|"
        r"(?:이렇게|이대로|이런\s*식).*?(?:끼|조이|고정|장착|연결).*?"
        r"(?:맞|괜찮|돼|될까|확인)|"
        r"(?:실제\s*)?(?:이미지|화면|카메라).*?(?:보고|봐서).*?"
        r"(?:조립|끼|맞|판단|확인)|"
        r"(?:잘|제대로|정상적으로).*?(?:했|됐|걸|끼|붙|조였)|"
        r"(?:잘|제대로|정상적으로).*?연결.*?(?:같|어때|보여|확인)|"
        r"연결한\s*것\s*같.*?(?:어때|보여|확인)|"
        r"(?:완료|끝|확인).*?(?:했|됐)?(?:나요|습니까|어요|어\?|죠)|"
        r"(?:이|그|지금)\s*(?:상태|정도).*?(?:맞|괜찮|정상)|"
        r"(?:확인|검사)해\s*(?:줘|주세요))", re.I)
    TARGET_STEP_INFO = re.compile(
        r"(?:(?:step|스텝)\s*[0-9일이삼사오육칠팔구십]+(?:\s*단계)?|"
        r"[0-9일이삼사오육칠팔구십]+\s*단계)\s*(?:작업|과정)?\s*"
        r"(?:은|는|이|가|을|를)?\s*(?:뭐|무엇|어떻게|방법|과정|설명|알려|안내|보여)", re.I)
    TARGET_STEP_PAGE = re.compile(
        r"[0-9일이삼사오육칠팔구십]+\s*단계\s*(?:의\s*)?"
        r"(?:페이지|자료|도면)(?:도|를|을)?\s*(?:보여|띄워|열어|바꿔|수정)", re.I)
    TARGET_STEP_REFERENCE = re.compile(
        r"[0-9일이삼사오육칠팔구십]+\s*단계\s*(?:의\s*)?"
        r"(?:참고\s*)?(?:화면|사진|이미지)(?:도|를|을)?\s*"
        r"(?:보여|띄워|열어|켜)", re.I)
    MANUAL_VIEW = re.compile(
        r"(?:pdf|피디에프|조립서|매뉴얼|메뉴얼).*?"
        r"(?:보여|띄워|열어|내용|목록|단계|페이지|확인할\s*수|확인\s*가능)|"
        r"(?:보여|띄워|열어).*?(?:pdf|피디에프|조립서|매뉴얼|메뉴얼)", re.I)

    # Physical fetch requests: an object word plus a bring/hand-over verb in
    # either order. The object vocabulary matches robot_skills' detector
    # aliases; unknown objects still route here so the engine can reply
    # honestly instead of chatting past a physical request.
    FETCH_OBJECT = re.compile(
        r"(?:해머|해먹|함마|함머|햄머|망치|드라이버|렌치|스패너|펜치|플라이어|드릴|공구|"
        r"hammer|screwdriver|wrench|pliers|drill)"
        r"[^.?!]*?(?:가져|갖다|챙겨|건네|집어|가지고\s*와|들고\s*와)|"
        r"(?:가져|갖다|챙겨|건네|집어)[^.?!]*?"
        r"(?:해머|해먹|함마|함머|햄머|망치|드라이버|렌치|스패너|펜치|플라이어|드릴|공구|"
        r"hammer|screwdriver|wrench|pliers|drill)", re.I)
    # Tidy-away: an object word plus a put-away verb. Checked BEFORE
    # FETCH_OBJECT because "갖다 놔" contains the fetch verb "갖다".
    TIDY_OBJECT = re.compile(
        r"(?:해머|해먹|함마|함머|햄머|망치|드라이버|렌치|스패너|펜치|플라이어|드릴|공구|"
        r"hammer|screwdriver|wrench|pliers|drill)"
        r"[^.?!]*?(?:정리|제자리|치워|갖다\s*(?:놔|둬|놓)|넣어\s*둬|보관)|"
        r"(?:정리|치워|갖다\s*(?:놔|둬|놓))[^.?!]*?"
        r"(?:해머|해먹|함마|함머|햄머|망치|드라이버|렌치|스패너|펜치|플라이어|드릴|공구)", re.I)
    # "지금 이거 잘하고 있는 거 맞아?" while pointing at the work. This must
    # be checked BEFORE the assembly-question routes: those answer from the
    # manual text, whereas this one sends the robot to LOOK. The giveaway is
    # a deictic ("이거", "이렇게", "지금") plus a doing-verb, with no object
    # name - if they name a tool they want it fetched, not inspected.
    #
    # A DEICTIC is required. "제가 잘했는지 확인해 주세요" asks about the work
    # in general and the assembly router answers it from the manual and the
    # fixed camera; "이거 맞아?" asks about the thing under the operator's
    # finger, and only a photograph of that spot can answer it. Dropping the
    # deictic requirement swallowed the first kind too.
    INSPECT_STEP = re.compile(
        r"(?:이거|이게|이렇게|이쪽|여기|지금|방금)[^.?!]{0,20}"
        r"(?:맞(?:아|나|는지|게)|제대로|잘\s*(?:하고|되고|된|한)|"
        r"괜찮(?:아|은지|나)|이상\s*(?:없|한)|확인(?:해|좀)?)|"
        r"(?:이거|이게|여기|이쪽)\s*[^.?!]{0,12}?"
        r"(?:봐\s*줘|봐줘|봐\s*주|한\s*번\s*보|보고\s*(?:말|알려)|좀\s*봐)|"
        r"(?:사진|카메라)(?:로)?\s*(?:찍어|확인)(?:\s*(?:줘|봐))?", re.I)
    #: "더 가까이서 봐줘" / "멀리서 다시" - a distance request that only makes
    #: sense as a follow-up to an inspection, so it rides the same intent and
    #: the engine reads the modifier out of the text.
    INSPECT_CLOSER = re.compile(
        r"(?:더|좀|조금)?\s*(?:가까이|가까이서|자세히|크게|확대)"
        r"[^.?!]{0,10}(?:봐|보고|찍|다시|확인)?", re.I)
    INSPECT_FARTHER = re.compile(
        r"(?:더|좀|조금)?\s*(?:멀리|멀리서|넓게|전체|주변까지)"
        r"[^.?!]{0,10}(?:봐|보고|찍|다시|확인)?", re.I)
    #: "손에 있는 거 갖다 놔" - take back what the operator is holding and
    #: put it down. Checked before the tidy/fetch rules because it names no
    #: tool: the object is whatever is on the hand right now.
    TAKE_FROM_HAND = re.compile(
        # "내 손 위에 있는 거 다시 갖다놔" - the possessive, the postposition
        # and the filler between the hand and the verb are all optional and
        # all showed up in real phrasing, so the gap is what is matched.
        r"(?:내|제|나의)?\s*손\s*(?:위|바닥)?\s*(?:에|에서|의)?\s*"
        r"[^.?!]{0,14}?"
        r"(?:갖다\s*(?:놔|놓)|갖다놔|가져다\s*(?:놔|놓)|가져다놔|"
        r"내려\s*(?:놔|놓)|내려놔|놓아|놔\s*줘|치워|가져가|받아)|"
        r"(?:이거|이것|저거|저것)\s*(?:좀\s*)?(?:다시\s*)?"
        r"(?:갖다\s*(?:놔|놓)|갖다놔|가져다\s*(?:놔|놓)|가져다놔|"
        r"내려\s*(?:놔|놓)|내려놔|치워|가져가|받아|놓고|놔\s*두고)", re.I)
    #: "저거 가져와" while pointing - the object is chosen by the finger, so
    #: no tool name appears and resolve_object_class would find nothing.
    FETCH_POINTED = re.compile(
        r"(?:저거|저것|이거|이것|그거|저\s*공구|이\s*공구)\s*(?:좀\s*)?"
        r"(?:가져|갖다|챙겨|건네|집어|줘|주라|줄래)", re.I)
    # --- paraphrase fallback ------------------------------------------
    # The patterns above catch the phrasings we have actually heard. People
    # do not repeat phrasings: "여기 와서 봐줘", "이 부분 좀 확인해줄래",
    # "저거 나한테 줘" all mean the same three jobs and only the first was
    # matched. Rather than growing the patterns forever, decide by the PARTS
    # a request is made of - what is being referred to, and what is being
    # asked - which is what survives rewording.
    _DEICTIC = re.compile(
        r"이거|이것|이건|저거|저것|저건|그거|그것|여기|저기|거기|이쪽|저쪽|"
        r"이\s*(?:부분|자리|곳|위치)|저\s*(?:공구|물건)", re.I)
    _OWN_HAND = re.compile(
        r"(?:내|제|나의)?\s*손(?:에|위|바닥|안)|들고\s*있는|쥐고\s*있는|"
        r"내가\s*든|잡고\s*있는", re.I)
    _LOOK = re.compile(
        r"봐|보고|보여|살펴|확인|체크|점검|검사|사진|찍어|어떤지|어때|"
        r"이상\s*(?:없|있)|괜찮", re.I)
    _JUDGE = re.compile(r"맞|제대로|잘\s*(?:하|되|됐|한)|올바|정상", re.I)
    _GIVE = re.compile(
        r"가져\s*와|가져와|갖다\s*줘|갖다줘|가져다\s*줘|집어|건네|"
        r"달라|주라|줄래|줘|줍|가지고\s*와", re.I)
    _PUTBACK = re.compile(
        r"갖다\s*(?:놔|놓)|가져다\s*(?:놔|놓)|내려\s*(?:놔|놓)|치워|"
        r"가져가|제자리|놔\s*(?:둬|줘)|놓아|반납", re.I)
    #: Looser put-down stems, used ONLY together with a connector and a
    #: fetch verb. "받고", "치우고", "놓고" are the conjugations that show up
    #: mid-sentence, where the strict forms above never appear.
    _PUTBACK_STEM = re.compile(
        r"갖다\s*(?:놔|놓)|가져다\s*(?:놔|놓)|내려\s*(?:놔|놓)|치우|치워|"
        r"가져가|제자리|반납|받고|받아|놓고|놔\s*두", re.I)
    #: Two jobs in one sentence need a joint - without one, "이거 좀 가져가"
    #: reads as both a put-back and a fetch and would run two motions.
    _AND_THEN = re.compile(r"놓고|놔두고|받고|주고|하고\s*(?:나서|난\s*뒤)?|"
                           r"그리고|그다음|그\s*다음|다음에|한\s*뒤", re.I)

    def fallback_robot_intent(self, text):
        """Robot intent from the parts of the request, or None.

        Deliberately conservative: no deictic and no reference to the
        operator's own hand means this is not one of these jobs, whatever
        verbs it contains. "가져와" alone is answered by the named-object
        route or by conversation, not by guessing at what "it" was."""
        deictic = bool(self._DEICTIC.search(text))
        own_hand = bool(self._OWN_HAND.search(text))
        if not (deictic or own_hand):
            return None
        give = bool(self._GIVE.search(text))
        putback = bool(self._PUTBACK.search(text))
        look = bool(self._LOOK.search(text))
        judge = bool(self._JUDGE.search(text))
        if putback and give and self._AND_THEN.search(text):
            return "TAKE_THEN_FETCH"
        if putback or (own_hand and not (look or give)):
            return "TAKE_FROM_HAND"
        if look or judge:
            return "INSPECT_STEP"
        if give:
            return "FETCH_POINTED"
        return None

    ROBOT_HOME = re.compile(
        r"(?:로봇|팔)?\s*(?:홈|home)\s*(?:포즈|위치|자세)?(?:으로|로)?\s*"
        r"(?:가|돌아가|이동|복귀)|로봇\s*제자리로", re.I)
    GRIPPER_OPEN = re.compile(r"그리퍼\s*(?:열어|벌려|open)", re.I)
    GRIPPER_CLOSE = re.compile(r"그리퍼\s*(?:닫아|접어|close)", re.I)
    # While the ROBOT is physically moving, a bare stop word is an immediate
    # physical stop - never a scope-clarification question.
    ROBOT_STOP = re.compile(r"(?:멈춰|정지|스톱|stop|그만)", re.I)

    # A question containing a step number and "작업" must not be mistaken for
    # a start command merely because START_WORK contains that noun sequence.
    QUESTION_ENDING = re.compile(
        r"(?:나요|까요|습니까|맞아|맞나요|했어\?|됐어\?|"
        r"인가요|인지|확인해|것\s*같|거\s*같|거야|"
        r"몇\s*단계(?:야|예요|인가))", re.I)

    #: Longest answer still treated as a bare yes/no. Past this the user is
    #: saying something substantive that deserves normal routing.
    CONFIRM_MAX_LEN = 22

    def confirmation_answer(self, text: str) -> str | None:
        """'ACCEPT' / 'REJECT' / None for an answer to a pending question."""
        value = " ".join(text.strip().split()).strip(" .,!?~")
        if not value:
            return None
        if self.CANCEL.fullmatch(value):
            return "REJECT"
        if self.CONFIRM.fullmatch(value):
            return "ACCEPT"
        if len(value) > self.CONFIRM_MAX_LEN:
            return None
        # "그거 말고 드라이버 가져와줘" names a different job: that is a new
        # request, not an answer, so let the normal router own it.
        if self.FETCH_OBJECT.search(value) or self.TIDY_OBJECT.search(value):
            return None
        if self.CONFIRM_NEGATIVE.search(value):
            return "REJECT"
        if self.CONFIRM_HEAD.match(value) or self.CONFIRM_VERB.match(value):
            return "ACCEPT"
        return None

    def route(self, text: str, *, has_pending_confirmation: bool = False,
              robot_skill_active: bool = False) -> UnifiedIntent:
        value = re.sub(r"에이\s*(?:알|어|열)", "AR", text.strip(), flags=re.I)
        # Safety first: while the robot is physically moving, any stop word
        # means STOP THE ROBOT, immediately, before every other reading.
        if robot_skill_active and self.ROBOT_STOP.search(value):
            return UnifiedIntent("ROBOT_SKILL", "STOP", value)
        # A question we just asked owns the next yes/no. "응, 가져와" answers
        # the pending fetch confirmation - reading it as a NEW fetch request
        # only re-asks the same question and the robot never starts.
        if has_pending_confirmation:
            answer = self.confirmation_answer(value)
            if answer is not None:
                return UnifiedIntent("CONFIRMATION", answer, value)
        # Pointing-and-asking beats every text-only assembly answer: the
        # operator is asking about the thing under their finger, which only a
        # photograph can settle.
        # "이거 가져다 놓고 저거 가져와" is two jobs in one breath, and
        # answering only the first half is the kind of literal-mindedness
        # that makes an assistant tiring to use.
        # Compound, decided from the parts: something of mine put down,
        # then something else fetched, with a joint between them. Checked
        # before the single-job routes so the second half is not lost.
        if (self._AND_THEN.search(value)
                and self._PUTBACK_STEM.search(value)
                and self._GIVE.search(value)
                and (self._OWN_HAND.search(value)
                     or self._DEICTIC.search(value))):
            return UnifiedIntent("ROBOT_SKILL", "TAKE_THEN_FETCH", value)
        if self.TAKE_FROM_HAND.search(value):
            return UnifiedIntent("ROBOT_SKILL", "TAKE_FROM_HAND", value)
        if (self.INSPECT_CLOSER.search(value)
                or self.INSPECT_FARTHER.search(value)) and (
                self.INSPECT_STEP.search(value)
                or re.search(r"봐|찍|확인|보여", value)):
            return UnifiedIntent("ROBOT_SKILL", "INSPECT_STEP", value)
        if (self.INSPECT_STEP.search(value)
                and not self.FETCH_OBJECT.search(value)
                and not self.TIDY_OBJECT.search(value)):
            return UnifiedIntent("ROBOT_SKILL", "INSPECT_STEP", value)
        if self.TIDY_OBJECT.search(value):
            return UnifiedIntent("ROBOT_SKILL", "TIDY_OBJECT", value)
        # "저거 가져와" names nothing - the finger already did.
        if self.FETCH_POINTED.search(value):
            return UnifiedIntent("ROBOT_SKILL", "FETCH_POINTED", value)
        if self.FETCH_OBJECT.search(value):
            return UnifiedIntent("ROBOT_SKILL", "FETCH_OBJECT", value)
        if self.GRIPPER_OPEN.search(value):
            return UnifiedIntent("ROBOT_SKILL", "GRIPPER_OPEN", value)
        if self.GRIPPER_CLOSE.search(value):
            return UnifiedIntent("ROBOT_SKILL", "GRIPPER_CLOSE", value)
        if self.ROBOT_HOME.search(value):
            return UnifiedIntent("ROBOT_SKILL", "HOME", value)
        if self.SAVE_PROGRESS_AND_STOP.search(value):
            return UnifiedIntent("SYSTEM", "SAVE_PROGRESS_AND_CLARIFY_STOP", value)
        if self.SAVE_PROGRESS_AND_HIDE.search(value):
            return UnifiedIntent("SYSTEM", "SAVE_PROGRESS_AND_DISPLAY_OFF", value)
        # Whole-copilot shutdown wins over a display clause in compound
        # commands such as "AR 화면 끄고 코파일럿 종료".
        if self.STOP_COPILOT.search(value):
            return UnifiedIntent("SYSTEM", "STOP_COPILOT", value)
        if self.HIDE_DISPLAY.search(value):
            return UnifiedIntent("SYSTEM", "DISPLAY_OFF", value)
        # A live RealSense view is a different operation from a manual
        # reference image. It must win even if the user says "reference, not".
        if self.SHOW_LIVE_DISPLAY.search(value):
            return UnifiedIntent("SYSTEM", "DISPLAY_LIVE", value)
        if self.CAMERA_FEEDBACK.search(value):
            return UnifiedIntent("SYSTEM", "CAMERA_FEEDBACK", value)
        # Questions or complaints about a previous display action must never
        # execute that action again merely because they repeat "show/image".
        if self.DISPLAY_FEEDBACK.search(value):
            return UnifiedIntent("SYSTEM", "DISPLAY_FEEDBACK", value)
        if self.VISION_FEEDBACK.search(value):
            return UnifiedIntent("SYSTEM", "VISION_FEEDBACK", value)
        if self.STEP_MAPPING_CHALLENGE.search(value):
            return UnifiedIntent("ASSEMBLY", "QUESTION", value)
        if self.RETURN_TO_STEP.search(value):
            return UnifiedIntent("ASSEMBLY", "QUESTION", value)
        if self.STATE_FEEDBACK.search(value):
            return UnifiedIntent("SYSTEM", "STATE_FEEDBACK", value)
        if self.TARGET_STEP_PAGE.search(value) or self.TARGET_STEP_REFERENCE.search(value):
            return UnifiedIntent("ASSEMBLY", "SHOW_LOADED_MANUAL", value)
        # Numbered step requests belong to the assembly router even when they
        # contain "사진 보여줘"; it can preserve combined explanation+image.
        if self.TARGET_STEP_INFO.search(value):
            return UnifiedIntent("ASSEMBLY", "TARGET_STEP_INFO", value)
        if self.SHOW_REFERENCE.search(value):
            return UnifiedIntent("SYSTEM", "SHOW_REFERENCE", value)
        if self.SHOW_DISPLAY.search(value):
            return UnifiedIntent("SYSTEM", "DISPLAY_ON", value)
        if self.PAUSE_WORK.search(value):
            return UnifiedIntent("SYSTEM", "PAUSE_WORK", value)
        if self.AMBIGUOUS_STOP.fullmatch(value):
            return UnifiedIntent("SYSTEM", "CLARIFY_STOP_SCOPE", value)
        if re.search(r"(?:종료|끝내|꺼|그만|멈춰)(?:\s*해)?"
                     r"(?:\s*(?:줘|주세요|줄래))?[.!?]?$|"
                     r"(?:종료|끝내|그만)할게[.!?]?$",
                     value, re.I):
            return UnifiedIntent("SYSTEM", "CLARIFY_STOP_SCOPE", value)
        if has_pending_confirmation and self.CANCEL.fullmatch(value):
            return UnifiedIntent("CONFIRMATION", "REJECT", value)
        if has_pending_confirmation and self.CONFIRM.fullmatch(value):
            return UnifiedIntent("CONFIRMATION", "ACCEPT", value)
        if self.MANUAL_VIEW.search(value):
            return UnifiedIntent("ASSEMBLY", "SHOW_LOADED_MANUAL", value)
        if self.MANUAL.search(value):
            return UnifiedIntent("MANUAL", "GENERATE_OR_SCAN", value)
        if self.PROGRESS_UPDATE.search(value):
            return UnifiedIntent("ASSEMBLY", "USER_PROGRESS_UPDATE", value)
        if self.ACTIVE_STEP_UPDATE.search(value):
            return UnifiedIntent("ASSEMBLY", "ACTIVE_STEP_UPDATE", value)
        if self.ASSEMBLY.search(value) and self.QUESTION_ENDING.search(value):
            return UnifiedIntent("ASSEMBLY", "QUESTION", value)
        if self.START_WORK.search(value):
            return UnifiedIntent("SYSTEM", "START_WORK", value)
        if self.ASSEMBLY.search(value):
            return UnifiedIntent("ASSEMBLY", "QUESTION", value)
        # Nothing above recognised it. Before handing a physical request to
        # small talk, decide by its parts.
        guessed = self.fallback_robot_intent(value)
        if guessed is not None:
            return UnifiedIntent("ROBOT_SKILL", guessed, value)
        if self.DESCRIBE.search(value):
            return UnifiedIntent("VISION", "DESCRIBE", value)
        if self.WORK_REPORT.search(value):
            return UnifiedIntent("ASSEMBLY", "USER_WORK_REPORT", value)
        if self.SEARCH.search(value):
            return UnifiedIntent("SEARCH", "WEB_SEARCH", value)
        return UnifiedIntent("CONVERSATION", "CHAT", value)


def extract_step_number(text: str) -> int | None:
    match = re.search(r"([0-9]+|[일이삼사오육칠팔구십]+)\s*단계", text)
    if match is None:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    digits = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
              "육": 6, "칠": 7, "팔": 8, "구": 9}
    if "십" in token:
        tens, ones = token.split("십", 1)
        return (digits.get(tens, 1) * 10) + digits.get(ones, 0)
    return digits.get(token)
