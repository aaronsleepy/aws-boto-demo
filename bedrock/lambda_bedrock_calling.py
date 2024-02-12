import json
import boto3

class BedrockClientDemo:
    """
    Bedrock Runtime 클라이언트를 생성하고 관리하기 위한 클래스
    Bedrock Runtime 클라이언트를 사용하여 Bedrock Claude 모델에 질문을 하는 기능을 제공

    클래스의 구현은 아래 블로그를 기반으로 하며, 간략한 리팩톨링을 반영함
    https://aws.plainenglish.io/unleashing-the-power-of-conversational-ai-a-seamless-guide-to-creating-an-amazon-lex-chatbot-with-68f2e854377c
    """

    REGION_NAME = "us-east-1"
    LEX_INTENT_FALLBACK = "FallbackIntent"
    
    def create_bedrock_client(self):
        """
        boto3를 사용하여 Bedrock Runtime 클라이언트를 생성

        반환값:
        - boto3.client: Bedrock Runtime 클라이언트.
        """
        bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.REGION_NAME
        )
        return bedrock

    def query_action(self, question, bedrock):
        """
        주어진 사용자 질문으로 Bedrock Claude 모델을 쿼리합니다.

        매개변수:
        - question (str): 사용자의 입력/질문.
        - bedrock (boto3.client): Bedrock Runtime 클라이언트.

        반환값:
        - dict: Bedrock 모델로부터의 결과.
        """
        prompt = question
        print(f"prompt: {prompt}")

        body = json.dumps({
            "inputText": prompt, 
            "textGenerationConfig":{
                "maxTokenCount": BedrockModelConfig.MAX_TOKENS_COUNT,
                "stopSequences": [],
                "temperature": BedrockModelConfig.TEMPERATURE,
                "topP": BedrockModelConfig.TOP_P
            }
        }) 


        modelId = "amazon.titan-text-express-v1"
        contentType = "application/json"
        accept = "*/*"
            
        response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
        result = json.loads(response.get("body").read())
        print(f"result: {result}")
        return result

    def handle_fallback(self, event):
        """
        FallbackIntent를 처리하기 위해 사용자 입력으로 Bedrock 모델을 쿼리.

        매개변수:
        - event (dict): Lex 세션에 대한 정보를 포함하는 AWS Lambda 이벤트.

        반환값:
        - dict: Lex의 반환값 형태로 구조화된 Bedrock 모델의 응답.
        """
        slots = event["sessionState"]["intent"]["slots"]
        intent = event["sessionState"]["intent"]["name"]
        bedrock = self.create_bedrock_client()
        question = event["inputTranscript"]
        result = self.query_action(question, bedrock)
        session_attributes = event["sessionState"]["sessionAttributes"]

        response = {
            "sessionState": {
                "dialogAction": {
                    "type": "Close",
                },
                "intent": {"name": intent, "slots": slots, "state": "Fulfilled"},
                "sessionAttributes": session_attributes,
            },
            "messages": [
                {"contentType": "PlainText", "content": result.get('results')[0].get('outputText')},
            ],
        }
        return response

def lambda_handler(event, context):
    """
    AWS Lambda 핸들러 함수

    매개변수:
    - event (dict): AWS Lambda 이벤트.
    - context (object): AWS Lambda 컨텍스트.

    반환값:
    - dict: Lex 응답.
    """
    demo = BedrockClientDemo()
    session_attributes = event.get("sessionState", {}).get("sessionAttributes", {})
    intent = event.get("sessionState", {}).get("intent", {}).get("name", "")
    if intent == "FallbackIntent":
        return demo.handle_fallback(event)

class BedrockModelConfig:
    """
    Bedrock Claude 모델의 구성을 위한 상수를 정의하는 클래스
    """
    MAX_TOKENS_COUNT = 500
    TEMPERATURE = 1
    TOP_K = 250
    TOP_P = 0.99