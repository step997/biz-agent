"""Guards：输入侧拦提示词注入 / 越权诉求 / 超长输入；输出侧脱敏手机号与身份证。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_LEN = 500

INJECTION_PATTERNS = [
    r"忽略(掉)?(之前|以上|上面|前面|所有|全部)?(的)?(所有|全部)?(指令|提示|设定|规则|要求)",
    r"ignore\s+(all\s+)?(previous|above)\s+(instructions?|prompts?)",
    r"(system|系统)\s*prompt",
    r"(打印|输出|显示|告诉我|复述)(一下)?(你的)?(系统)?(提示词|指令|设定)",
    r"你现在(是|扮演)",
    r"开发者模式|developer\s*mode|jailbreak|越狱",
    r"you\s+are\s+now\s+",
]

# 越权 / 高风险诉求：对外一律走人工，不让智能体自己决定
SENSITIVE_PATTERNS = [
    (r"(导出|下载|拉取)(全部|所有|整个)(客户|用户|订单|手机号|数据库)", "批量导出客户数据"),
    (r"(删除|清空|drop|truncate)\s*(掉)?\s*(全部|所有|数据库|表)", "批量删除数据"),
    (r"(改|调整|篡改)(一下)?(数据库|后台|订单金额)", "直接改库"),
    (r"(绕过|跳过)(审批|风控|权限)", "绕过审批"),
    (r"(身份证|银行卡|密码|验证码)(号|信息)?.*(发我|给我|列出|查询)", "敏感字段查询"),
]

PHONE = re.compile(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)")
IDCARD = re.compile(r"(?<!\d)(\d{4})\d{10}(\d{3}[\dXx])(?!\d)")


@dataclass
class GuardResult:
    allow: bool
    reason: str = ""
    flags: list[str] = field(default_factory=list)
    text: str = ""


def check_input(question: str) -> GuardResult:
    text = (question or "").strip()
    flags: list[str] = []
    if not text:
        return GuardResult(allow=False, reason="问题是空的，请补充要查什么。", flags=["empty"], text=text)
    if len(text) > MAX_LEN:
        return GuardResult(allow=False, reason=f"问题过长（>{MAX_LEN} 字），请拆成具体的一条。",
                           flags=["too_long"], text=text[:MAX_LEN])

    for pat in INJECTION_PATTERNS:
        if re.search(pat, text, flags=re.I):
            flags.append("prompt_injection")
            return GuardResult(
                allow=False,
                reason="检测到试图改写系统设定的内容，本次请求已拦截并记录审计。",
                flags=flags, text=text)

    for pat, label in SENSITIVE_PATTERNS:
        if re.search(pat, text, flags=re.I):
            flags.append(f"sensitive:{label}")
            return GuardResult(
                allow=False,
                reason=f"「{label}」属于人工操作范围，智能体只提供查询与建单能力，请走人工审批或后台操作。",
                flags=flags, text=text)

    return GuardResult(allow=True, flags=flags, text=text)


def mask_output(text: str) -> str:
    """手机号/身份证打码，避免智能体把完整 PII 直接吐到对话里。"""
    text = PHONE.sub(r"\1****\2", text)
    text = IDCARD.sub(r"\1**********\2", text)
    return text
