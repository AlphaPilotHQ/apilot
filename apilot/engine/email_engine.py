import smtplib
import traceback
from queue import Empty, Queue
from threading import Thread
from email.message import EmailMessage

from apilot.core.engine import BaseEngine, MainEngine
from apilot.core.event import EventEngine
from apilot.core.setting import SETTINGS


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
            receiver: str = SETTINGS["email.receiver"]

        msg: EmailMessage = EmailMessage()
        msg["From"] = SETTINGS["email.sender"]
        msg["To"] = receiver
        msg["Subject"] = subject
        msg.set_content(content)

        self.queue.put(msg)

    def run(self) -> None:
        """Main thread function to process email queue."""
        server: str = SETTINGS["email.server"]
        port: int = SETTINGS["email.port"]
        username: str = SETTINGS["email.username"]
        password: str = SETTINGS["email.password"]

        while self.active:
            try:
                msg: EmailMessage = self.queue.get(block=True, timeout=1)

                try:
                    with smtplib.SMTP_SSL(server, port) as smtp:
                        smtp.login(username, password)
                        smtp.send_message(msg)
                except Exception:
                    msg: str = f"邮件发送失败: {traceback.format_exc()}"
                    self.main_engine.log_error(msg, "EMAIL")
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
