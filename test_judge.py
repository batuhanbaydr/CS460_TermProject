from models import get_judge_model

judge = get_judge_model()

response = judge.invoke("Say hello in one short sentence.")

print(response.content)