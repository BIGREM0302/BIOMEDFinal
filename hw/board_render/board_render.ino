#include <FastLED.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// --- LED 設定 ---
#define LED_PIN     6
#define NUM_LEDS    59  // 42(棋盤) + 7(指示燈) + 10(充能條)
#define BRIGHTNESS  60  // 亮度 (0-255)，測試時先調低
#define LED_TYPE    WS2812B
#define COLOR_ORDER GRB

CRGB leds[NUM_LEDS];

// --- 顏色定義 ---
CRGB colorEmpty  = CRGB(5, 5, 5);      // 微微的暗灰(代表空位)
CRGB colorYellow = CRGB(255, 255, 0);  // 玩家(黃)
CRGB colorRed    = CRGB(255, 0, 100);  // AI(紅/粉)
CRGB colorCyan   = CRGB(0, 255, 255);  // 指示燈/AI提示
CRGB colorGreen  = CRGB(0, 255, 0);    // 充能條

// --- LCD 設定 ---
LiquidCrystal_I2C lcd(0x27, 16, 2); // 0x27 為多數 I2C LCD 預設位址

// --- 系統變數 ---
const byte numChars = 64;
char receivedChars[numChars];
boolean newData = false;

// 儲存解析後的狀態
char boardState[43]; 
int focusLevel = 0;
int indicatorCol = 0;
int aiHintCol = -1;
float redWinProb = 0.5;
float ylwWinProb = 0.5;

void setup() {
    Serial.begin(115200);
    
    // 初始化 LED
    FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS).setCorrection(TypicalLEDStrip);
    FastLED.setBrightness(BRIGHTNESS);
    FastLED.clear();
    FastLED.show();

    // 初始化 LCD
    lcd.init();
    lcd.backlight();
    lcd.setCursor(0, 0);
    lcd.print("BCI Connect 4");
    lcd.setCursor(0, 1);
    lcd.print("System Ready...");
}

void loop() {
    recvWithStartEndMarkers();
    if (newData == true) {
        parseData();
        updateBoard();
        updateLCD();
        newData = false;
    }
}

// 讀取 <...> 封包
void recvWithStartEndMarkers() {
    static boolean recvInProgress = false;
    static byte ndx = 0;
    char startMarker = '<';
    char endMarker = '>';
    char rc;

    while (Serial.available() > 0 && newData == false) {
        rc = Serial.read();

        if (recvInProgress == true) {
            if (rc != endMarker) {
                receivedChars[ndx] = rc;
                ndx++;
                if (ndx >= numChars) {
                    ndx = numChars - 1;
                }
            } else {
                receivedChars[ndx] = '\0'; // 結束字串
                recvInProgress = false;
                ndx = 0;
                newData = true;
            }
        } else if (rc == startMarker) {
            recvInProgress = true;
        }
    }
}

// 解析逗號分隔的資料
void parseData() {
    char * strtokIndx; 
    
    strtokIndx = strtok(receivedChars, ",");
    strcpy(boardState, strtokIndx);
    
    strtokIndx = strtok(NULL, ",");
    focusLevel = atoi(strtokIndx);
    
    strtokIndx = strtok(NULL, ",");
    indicatorCol = atoi(strtokIndx);
    
    strtokIndx = strtok(NULL, ",");
    aiHintCol = atoi(strtokIndx);
    
    strtokIndx = strtok(NULL, ",");
    redWinProb = atof(strtokIndx);
    
    strtokIndx = strtok(NULL, ",");
    ylwWinProb = atof(strtokIndx);
}

// 根據接線順序映射 LED (這裡假設串聯順序為：棋盤 0~41 -> 指示燈 42~48 -> 充能條 49~58)
void updateBoard() {
    FastLED.clear();

    // 1. 繪製 7x6 棋盤 (索引 0~41)
    for (int y = 0; y < 6; y++) {
        for (int x = 0; x < 7; x++) {
            int stringIndex = y * 7 + x; // 對應 Python 送來的字串索引
            
            // 將二維座標轉成 LED 燈條實際的索引
            // ⚠️ 依據你黏貼燈條的方向 (蛇行或單向)，這裡的 mapping 需要微調。
            // 這裡假設是左上到右下的單向接法：
            int ledIndex = stringIndex; 
            
            if (boardState[stringIndex] == '0') {
                leds[ledIndex] = colorEmpty;
            } else if (boardState[stringIndex] == '1') {
                leds[ledIndex] = colorYellow;
            } else if (boardState[stringIndex] == '2') {
                leds[ledIndex] = colorRed;
            }
        }
    }

    // 2. 繪製頂部指示燈 (索引 42~48)
    int indicatorBaseIdx = 42;
    // 使用 millis() 製造 AI 提示閃爍效果 (每 200ms 切換)
    bool flashOn = (millis() % 400) < 200;

    for (int i = 0; i < 7; i++) {
        if (i == aiHintCol && flashOn) {
            leds[indicatorBaseIdx + i] = colorCyan; // 雷射提示大絕招
        } else if (i == indicatorCol && aiHintCol == -1) {
            leds[indicatorBaseIdx + i] = colorYellow; // 玩家選擇指示燈
        } else {
            leds[indicatorBaseIdx + i] = CRGB::Black;
        }
    }

    // 3. 繪製充能條 (索引 49~58)
    int chargeBaseIdx = 49;
    int numChargeLeds = map(focusLevel, 0, 100, 0, 10); // 將 0-100% 對應到 10 顆 LED
    
    for (int i = 0; i < 10; i++) {
        if (i < numChargeLeds) {
            leds[chargeBaseIdx + i] = colorGreen;
        } else {
            leds[chargeBaseIdx + i] = CRGB(10, 10, 10); // 未充能部分微亮
        }
    }

    FastLED.show();
}

void updateLCD() {
    // 限制 LCD 更新頻率，避免畫面閃爍
    static unsigned long lastLcdUpdate = 0;
    if (millis() - lastLcdUpdate > 500) { 
        lcd.setCursor(0, 0);
        lcd.print("YLW(You): ");
        lcd.print(ylwWinProb * 100, 1);
        lcd.print("%  "); // 清除尾部殘影

        lcd.setCursor(0, 1);
        lcd.print("RED(AI) : ");
        lcd.print(redWinProb * 100, 1);
        lcd.print("%  ");
        
        lastLcdUpdate = millis();
    }
}