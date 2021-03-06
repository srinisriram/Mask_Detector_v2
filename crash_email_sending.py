import smtplib
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
import argparse
from Production.constants import *


class CrashReport:
    """
        This class composes and sends an email when the program crashes.
    """

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-l", "--log_file", type=str,
                            help="Provide the log file to send via email", default=LOG_FILE)
        parser.add_argument('-o', '--occupancy', type=bool,
                            help='Required to send occupancy crash email', default=False)
        parser.add_argument('-m', '--mask', type=bool,
                            help='Required to send mask crash email', default=True)
        self.args = parser.parse_args()

    def perform_job(self):
        self.email_send()

    def email_send(self):
        """
        This method sends an email
        :return:
        """
        msg = MIMEMultipart()
        sender_email = "maskdetector101@gmail.com"
        receiver_email = "adityaanand.muz@gmail.com, srinivassriram06@gmail.com, raja.muz@gmail.com, " \
                         "abhisar.muz@gmail.com, ssriram.78@gmail.com"
        password = "LearnIOT06!"
        msg['From'] = 'maskdetector101@gmail.com'
        msg['To'] = "adityaanand.muz@gmail.com, srinivassriram06@gmail.com, raja.muz@gmail.com, " \
                    "abhisar.muz@gmail.com, ssriram.78@gmail.com "
        msg['Date'] = formatdate(localtime=True)
        if self.args.occupancy:
            msg['Subject'] = 'The Occupancy Tracker has crashed!'
            body = "The Occupancy Tracker has crashed and the occupation log is attached below."
        elif self.args.mask:
            msg['Subject'] = 'The Mask Detector has crashed!'
            body = "The Mask Detector has crashed and the mask log is attached below"

        part = MIMEBase('application', "octet-stream")
        part.set_payload(open(self.args.log_file, "rb").read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment', filename=self.args.log_file)
        msg.attach(part)

        msg.attach(MIMEText(body, "plain"))
        context = ssl.create_default_context()
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email.split(","), msg.as_string())
        except Exception as e:
            print(type(e).__name__ + ': ' + str(e))
        else:
            email_sent_status = True
        finally:
            return email_sent_status


if __name__ == '__main__':
    CrashReport().perform_job()