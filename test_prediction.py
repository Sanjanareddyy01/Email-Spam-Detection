from utils import predict_email


email = """
Congratulations!
You have won a free prize.
Click now to claim your reward.
"""


result = predict_email(email)


print(result)