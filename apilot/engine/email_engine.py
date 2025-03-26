"""
邮件引擎模块

提供系统邮件发送功能，支持异步发送和错误处理
"""

import smtplib
import traceback
from email.message import EmailMessage
from queue import Empty, Queue
from threading import Thread
from typing import Dict, Any

from apilot.core import (
    BaseEngine,
    EventEngine,
    MainEngine
)
from apilot.utils.logger import get_logger

# Email default configuration
EMAIL_CONFIG: Dict[str, Any] = {
    "active": False,  # Default: email notification disabled
    "server": "smtp.gmail.com",
    "port": 587,
    "username": "",
    "password": "",  # Use app-specific password
    "sender": "",
    "receiver": ""
}

# 模块级初始化日志器
logger = get_logger("EmailEngine")


class EmailEngine(BaseEngine):
    """
    Provides email sending function.
    """

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """Initialize email engine."""
        super(EmailEngine, self).__init__(main_engine, event_engine, "email")

        self.thread: Thread = Thread(target=self.run)
        self.queue: Queue = Queue()
        self.active: bool = False

        self.main_engine.send_email = self.send_email

    def send_email(self, subject: str, content: str, receiver: str = "") -> None:
        """Send email with given subject and content to receiver."""
        # Start email engine when sending first email.
        if not self.active:
            self.start()

        # Use default receiver if not specified.
        if not receiver:
            receiver: str = EMAIL_CONFIG["receiver"]

        msg: EmailMessage = EmailMessage()
        msg["From"] = EMAIL_CONFIG["sender"]
        msg["To"] = receiver
        msg["Subject"] = subject
        msg.set_content(content)

        self.queue.put(msg)

    def run(self) -> None:
        """Main thread function to process email queue."""
        server: str = EMAIL_CONFIG["server"]
        port: int = EMAIL_CONFIG["port"]
        username: str = EMAIL_CONFIG["username"]
        password: str = EMAIL_CONFIG["password"]

        while self.active:
            try:
                msg: EmailMessage = self.queue.get(block=True, timeout=1)

                try:
                    with smtplib.SMTP_SSL(server, port) as smtp:
                        smtp.login(username, password)
                        smtp.send_message(msg)
                except Exception:
                    msg: str = f"邮件发送失败: {traceback.format_exc()}"
                    logger.error(msg)
            except Empty:
                pass

    def start(self) -> None:
        """Start the email engine thread."""
        self.active = True
        self.thread.start()

    def close(self) -> None:
        """Stop the email engine thread."""
        if not self.active:
            return

        self.active = False
        self.thread.join()
