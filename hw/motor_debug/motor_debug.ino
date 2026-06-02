const int motorPin = 11;

void setup() {
  pinMode(motorPin, OUTPUT);
}

void loop() {
  digitalWrite(motorPin, HIGH); // 震動
  delay(500);

  digitalWrite(motorPin, LOW);  // 停止
  delay(2500);
}