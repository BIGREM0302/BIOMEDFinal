#include <FastLED.h>

// 定義接線腳位與燈珠數量
#define DATA_PIN    8
#define NUM_LEDS    15

// 建立一個陣列來儲存每一顆 LED 的顏色資料
CRGB leds[NUM_LEDS];

void setup() {
  // 告訴 FastLED 燈條的晶片型號 (WS2812)、資料腳位與顏色編碼順序 (通常是 GRB)
  FastLED.addLeds<WS2812, DATA_PIN, GRB>(leds, NUM_LEDS);
  
  // 【最重要的一行：硬派安全鎖】
  // 設定供電電壓為 5V，最大允許電流為 400 毫安培 (mA)
  // 有了這行，不管你下面寫什麼超耗電的全亮特效，FastLED 都會強行把電流壓制在安全範圍內！
  FastLED.setMaxPowerInVoltsAndMilliamps(5, 400); 

  // (可選) 也可以設定一個全局基礎亮度 (0-255)
  FastLED.setBrightness(50);
}

void loop() {
  // 測試 1：紅色流水燈
  for(int i = 0; i < NUM_LEDS; i++) {
    leds[i] = CRGB::Blue;       // 將第 i 顆設為純紅色
    FastLED.show();            // 更新燈條
    delay(50);
    leds[i] = CRGB::Black;     // 關閉 (黑色)
  }

  // 測試 2：FastLED 內建的超炫彩虹漸層 (這就是 FastLED 強大的地方，數學運算極快)
  // 參數：(燈條陣列, 數量, 起始色相, 每顆燈的色相差)
  fill_rainbow(leds, NUM_LEDS, 0, 10); 
  FastLED.show();
  delay(2000);                 // 讓彩虹光停留兩秒鐘
  
  // 清除所有燈光，準備下一次迴圈
  FastLED.clear(); 
  FastLED.show();
  delay(500);
}