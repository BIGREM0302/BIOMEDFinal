#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// 常見 LCD I2C 位址
LiquidCrystal_I2C lcd27(0x27, 16, 2);
LiquidCrystal_I2C lcd3F(0x3F, 16, 2);

byte foundAddress = 0x00;

void scanI2C() {
  Serial.println("Scanning I2C bus...");

  int nDevices = 0;

  for (byte address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    byte error = Wire.endTransmission();

    if (error == 0) {
      Serial.print("I2C device found at 0x");

      if (address < 16) {
        Serial.print("0");
      }

      Serial.println(address, HEX);

      foundAddress = address;
      nDevices++;
    }
  }

  if (nDevices == 0) {
    Serial.println("No I2C devices found.");
  } else {
    Serial.print("Total devices found: ");
    Serial.println(nDevices);
  }

  Serial.println();
}

void testLCD_0x27() {
  Serial.println("Trying LCD at 0x27...");

  lcd27.init();
  lcd27.backlight();
  lcd27.clear();

  lcd27.setCursor(0, 0);
  lcd27.print("LCD 0x27 OK?");
  lcd27.setCursor(0, 1);
  lcd27.print("Hello Arduino");

  delay(2000);
}

void testLCD_0x3F() {
  Serial.println("Trying LCD at 0x3F...");

  lcd3F.init();
  lcd3F.backlight();
  lcd3F.clear();

  lcd3F.setCursor(0, 0);
  lcd3F.print("LCD 0x3F OK?");
  lcd3F.setCursor(0, 1);
  lcd3F.print("Hello Arduino");

  delay(2000);
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  delay(1000);

  Serial.println("================================");
  Serial.println("Arduino Uno I2C LCD Debug");
  Serial.println("SDA = A4 or SDA pin");
  Serial.println("SCL = A5 or SCL pin");
  Serial.println("Baudrate = 115200");
  Serial.println("================================");

  scanI2C();

  if (foundAddress == 0x27) {
    testLCD_0x27();
  } else if (foundAddress == 0x3F) {
    testLCD_0x3F();
  } else if (foundAddress != 0x00) {
    Serial.print("Found unknown I2C address: 0x");
    if (foundAddress < 16) Serial.print("0");
    Serial.println(foundAddress, HEX);
    Serial.println("You may need to change LiquidCrystal_I2C address manually.");

    // 還是嘗試兩個常見位址
    testLCD_0x27();
    testLCD_0x3F();
  } else {
    Serial.println("LCD not detected on I2C bus.");
    Serial.println("Check VCC, GND, SDA, SCL wiring.");
  }
}

void loop() {
  static unsigned long lastUpdate = 0;
  static int counter = 0;

  if (millis() - lastUpdate >= 1000) {
    lastUpdate = millis();

    Serial.print("Loop counter = ");
    Serial.println(counter);

    if (foundAddress == 0x27) {
      lcd27.clear();
      lcd27.setCursor(0, 0);
      lcd27.print("Addr: 0x27");
      lcd27.setCursor(0, 1);
      lcd27.print("Sec: ");
      lcd27.print(counter);
    } else if (foundAddress == 0x3F) {
      lcd3F.clear();
      lcd3F.setCursor(0, 0);
      lcd3F.print("Addr: 0x3F");
      lcd3F.setCursor(0, 1);
      lcd3F.print("Sec: ");
      lcd3F.print(counter);
    }

    counter++;
  }
}