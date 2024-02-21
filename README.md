# TrainingLLM

### Categories
1. **Fine-tuning Pretrained LLMs**: Adjusting pre-trained LLMs for a particular tasks by training them further on a specialized dataset.
2. **Training Foundation Models**: Where I train my own foundation model from scratch.

## Fine-tuning conversation examples
### ChatBuddy
User: Can you help me pick up my kids after school today? I'll need to run to a dentist appointment.
Assistant: Of course. Just give me the name of the school, and I'll pick them up.
User: The name is ABC Secondary school.
Assistant: I'll pick them up and bring them to your house.
User: Thanks buddy, I'll buy you a drink tonight.
Assistant: I'm on the wagon, but I'll take a soda.

### SQL
Question: What is the highest age of users with name Danjie
Context: CREATE TABLE user (age INTEGER, name STRING)
Answer: SELECT MAX(age) FROM user WHERE name = "Danjie"

## Foundation model
Training LLM foundation models with Llama2 architecture.
