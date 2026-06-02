#include <FastLED.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// --- LED 設定 ---
#define LED_PIN     6
#define NUM_LEDS    49      
#define BRIGHTNESS  60
#define RING_BRIGHTNESS 25   // 勝率環亮度，0~255，越小越暗  
#define LED_TYPE    WS2812B
#define COLOR_ORDER GRB

// --- 充能條設定 ---
#define FOCUS_PIN       7       
#define NUM_FOCUS_LEDS  16      
CRGB focusLeds[NUM_FOCUS_LEDS]; 

// --- 【新增】勝率環形燈設定 ---
#define RING_PIN        8       // 將環形燈接在 Pin 8
#define NUM_RING_LEDS   24      // 環形燈總數 24 顆
CRGB ringLeds[NUM_RING_LEDS];   // 建立環形燈專用的色彩陣列

// --- 震動模組設定 ---
#define VIB_PIN 9 // 假設接到 Digital Pin 9，請依據實際接線修改
int lastPiecesCount = -1;        // 記錄上一次的棋子總數
unsigned long vibStartTime = 0;  // 記錄震動開始時間
bool isVibrating = false;        // 震動狀態旗標

CRGB leds[NUM_LEDS];

// --- 顏色定義 ---
CRGB colorEmpty  = CRGB(0, 0, 0);      
CRGB colorYellow = CRGB(255, 255, 0);
CRGB colorRed    = CRGB(255, 0, 100);  
CRGB colorCyan   = CRGB(0, 255, 255);
CRGB colorGreen  = CRGB(0, 255, 0);    
CRGB colorIndicator = CRGB(255, 255, 255);

// --- LCD 設定 ---
LiquidCrystal_I2C lcd(0x27, 16, 2);

// --- 系統變數 ---
const byte numChars = 64;
char receivedChars[numChars];
boolean newData = false;

int uiState = 0;
char boardState[43];
int focusLevel = 0;
int indicatorCol = 0;
int aiHintCol = -1;
float redWinProb = 0.5;
float ylwWinProb = 0.5;

void setup() {
    Serial.begin(115200);
    
    FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS).setCorrection(TypicalLEDStrip);
    FastLED.addLeds<LED_TYPE, FOCUS_PIN, COLOR_ORDER>(focusLeds, NUM_FOCUS_LEDS).setCorrection(TypicalLEDStrip);
    // 【新增】註冊環形燈條
    FastLED.addLeds<LED_TYPE, RING_PIN, COLOR_ORDER>(ringLeds, NUM_RING_LEDS).setCorrection(TypicalLEDStrip);

    FastLED.setBrightness(BRIGHTNESS);
    FastLED.clear();
    FastLED.show();

    // 初始化震動腳位
    pinMode(VIB_PIN, OUTPUT);
    digitalWrite(VIB_PIN, LOW);

    lcd.init();
    lcd.backlight();
    lcd.setCursor(0, 0);
    lcd.print("SYSTEM BOOTING..");
}

void loop() {
    recvWithStartEndMarkers();
    if (newData == true) {
        parseData();
        updateBoard();
        updateLCD();
        newData = false;
    }

    // 非阻塞式震動控制：如果正在震動，且已經過了 200 毫秒，就關閉震動
    if (isVibrating && (millis() - vibStartTime >= 200)) {
        isVibrating = false;
        digitalWrite(VIB_PIN, LOW);
    }
}

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
                receivedChars[ndx] = '\0';
                recvInProgress = false;
                ndx = 0;
                newData = true;
            }
        } else if (rc == startMarker) {
            recvInProgress = true;
        }
    }
}

void parseData() {
    char * strtokIndx;
    
    strtokIndx = strtok(receivedChars, ",");
    uiState = atoi(strtokIndx);
    
    strtokIndx = strtok(NULL, ",");
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

    // --- 新增：震動提醒邏輯 ---
    if (uiState == 2) { // 確保在遊戲進行中才震動
        int currentPieces = 0;
        
        // 計算盤面上的總棋子數 (不等於 '0' 就是有下棋)
        for (int i = 0; i < 42; i++) {
            if (boardState[i] != '0') {
                currentPieces++;
            }
        }
        
        // 如果棋子數量改變了 (代表剛有人下完一步，或是遊戲剛開始)
        if (currentPieces != lastPiecesCount) {
            // 偶數顆棋子代表輪到黃方 (玩家)
            if (currentPieces % 2 == 0) {
                isVibrating = true;
                vibStartTime = millis();
                digitalWrite(VIB_PIN, HIGH);
            }
            lastPiecesCount = currentPieces;
        }
    } else {
        // 如果不在遊戲中 (例如待機或結算畫面)，重置狀態確保下一局能正常觸發
        lastPiecesCount = -1;
    }
}

// 蛇行接線座標轉換 (0是最底層，往上遞增)
int getLedIndex(int x, int y) {
    int physicalRow = 5 - y;
    if (physicalRow % 2 == 0) {
        return (physicalRow * 7) + x;
    } else {
        return (physicalRow * 7) + (6 - x);
    }
}

void updateBoard() {
    FastLED.clear();

    if (uiState == 0) {
        fill_solid(leds, NUM_LEDS, CRGB::Red);
    } 
    else if (uiState == 1) {
        fill_solid(leds, NUM_LEDS, CRGB::Green);
    } 
    else {
        // [遊戲/結算 狀態]
        bool flashOn = (millis() % 400) < 200;
        
        // --- 預先計算 AI 提示應該落下的精確 Y 座標 ---
        int targetY = -1;
        if (aiHintCol >= 0 && aiHintCol < 7) {
            for (int scanY = 5; scanY >= 0; scanY--) {
                if (boardState[scanY * 7 + aiHintCol] == '0') {
                    targetY = scanY;
                    break;
                }
            }
        }

        // --- 繪製棋盤 (7x6) ---
        for (int y = 0; y < 6; y++) {
            for (int x = 0; x < 7; x++) {
                int stringIndex = y * 7 + x;
                int ledIndex = getLedIndex(x, y); 
                
                if (ledIndex < NUM_LEDS) {
                    if (x == aiHintCol && y == targetY && flashOn) {
                        leds[ledIndex] = colorCyan;
                    } 
                    else if (boardState[stringIndex] == '0') {
                        leds[ledIndex] = colorEmpty;
                    } else if (boardState[stringIndex] == '1') {
                        leds[ledIndex] = colorYellow;
                    } else if (boardState[stringIndex] == '2') {
                        leds[ledIndex] = colorRed;
                    }
                }
            }
        }

        // --- 繪製頂部指示燈 (Index 42 ~ 48) ---
        if (uiState == 2) {
            // 正常遊戲中的游標
            for (int i = 0; i < 7; i++) {
                int indIndex = 42 + i;
                if (indIndex < NUM_LEDS) {
                    if (i == indicatorCol) {
                        leds[indIndex] = colorIndicator;
                    } else {
                        leds[indIndex] = colorEmpty;
                    }
                }
            }
        } else if (uiState >= 4 && uiState <= 6) {
            // --- 獲勝特效：來回掃描流動 (Knight Rider 特效) ---
            CRGB winColor;
            if (uiState == 4) winColor = colorYellow;       // 玩家贏
            else if (uiState == 5) winColor = colorRed;     // AI贏
            else winColor = CRGB(200, 200, 200);            // 平手(白色/淺灰)

            // 計算 0~13 的循環，達成來回掃描 (0~6 往右，7~13 往左)
            int flowIndex = (millis() / 70) % 14; 
            int litPos = flowIndex;
            if (litPos > 6) {
                litPos = 13 - litPos; // 反向計算
            }

            for (int i = 0; i < 7; i++) {
                int indIndex = 42 + i;
                if (indIndex < NUM_LEDS) {
                    if (i == litPos) {
                        leds[indIndex] = winColor; // 最亮點
                    } else if (abs(i - litPos) == 1) {
                        leds[indIndex] = winColor;
                        leds[indIndex].fadeToBlackBy(200); // 兩側殘影 (調暗)
                    } else {
                        leds[indIndex] = colorEmpty;
                    }
                }
            }
        }

        // --- 繪製類比連續充能條 ---
        float fillAmount = (focusLevel / 100.0) * NUM_FOCUS_LEDS;
        for (int i = 0; i < NUM_FOCUS_LEDS; i++) {
            uint8_t hue = map(i, 0, NUM_FOCUS_LEDS, 96, 0);
            if (i < (int)fillAmount) {
                focusLeds[i] = CHSV(hue, 255, 255);
            } 
            else if (i == (int)fillAmount) {
                float fraction = fillAmount - (int)fillAmount;
                focusLeds[i] = CHSV(hue, 255, fraction * 255);
            } 
            else {
                focusLeds[i] = CRGB::Black;
            }
        }
    }

    // --- 【新增】繪製勝率環形燈 ---
    if (uiState == 0 || uiState == 1) {
        fill_solid(ringLeds, NUM_RING_LEDS, CRGB::Black); // 待機/規則時保持熄滅
    } else {
        // 計算紅色方(AI)應該佔據的顆數
        // 例: redWinProb=0.5 -> round(0.5 * 24) = 12顆紅色
        int redCount = round(redWinProb * NUM_RING_LEDS);
        
        for (int i = 0; i < NUM_RING_LEDS; i++) {
            if (i < redCount) {
                ringLeds[i] = colorRed;
            } else {
                ringLeds[i] = colorYellow;
            }

            // 單獨降低勝率環亮度
            ringLeds[i].nscale8(RING_BRIGHTNESS);
        }
    }

    FastLED.show();
}

void updateLCD() {
    static unsigned long lastLcdUpdate = 0;
    if (millis() - lastLcdUpdate > 500) { 
        lcd.clear();
        
        if (uiState == 0) {
            lcd.setCursor(0, 0);
            lcd.print("NEURAL CONNECT 4");
            lcd.setCursor(0, 1); lcd.print(" Blink To Start ");
        } 
        else if (uiState == 1) {
            lcd.setCursor(0, 0);
            lcd.print("SYSTEM PROTOCOLS");
            lcd.setCursor(0, 1); lcd.print(" Blink To Agree ");
        } 
        else if (uiState == 2) {
            lcd.setCursor(0, 0);
            lcd.print("YLW(You): "); lcd.print(ylwWinProb * 100, 1); lcd.print("%");
            lcd.setCursor(0, 1);
            lcd.print("RED(AI) : "); lcd.print(redWinProb * 100, 1); lcd.print("%");
        } 
        // --- 新增：結算停留在棋盤時的 LCD 顯示 ---
        else if (uiState == 4) {
            lcd.setCursor(0, 0); lcd.print("  YLW VICTORY!  ");
            lcd.setCursor(0, 1); lcd.print("  * YOU WIN * ");
        }
        else if (uiState == 5) {
            lcd.setCursor(0, 0); lcd.print("  RED VICTORY!  ");
            lcd.setCursor(0, 1); lcd.print("  * AI WINS * ");
        }
        else if (uiState == 6) {
            lcd.setCursor(0, 0); lcd.print("   STALEMATE!   ");
            lcd.setCursor(0, 1); lcd.print(" * NO WINNER * ");
        }
        // ---------------------------------------
        else if (uiState == 3) {
            lcd.setCursor(0, 0);
            lcd.print("MATCH TERMINATED");
            lcd.setCursor(0, 1); lcd.print("Rst/Quit ?");
        }
        
        lastLcdUpdate = millis();
    }
}