#include <LedControl.h>

// LedControl(DIN, CLK, CS, number_of_devices)
LedControl lc = LedControl(11, 13, 10, 1);

const int WIDTH = 8;
const int HEIGHT = 8;

bool board[HEIGHT][WIDTH];

int fallingX = 0;
int fallingY = 0;

void clearBoard() {
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      board[y][x] = false;
    }
  }
}

bool isBoardFull() {
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      if (!board[y][x]) {
        return false;
      }
    }
  }
  return true;
}

void drawBoard() {
  lc.clearDisplay(0);

  // 畫已經累積的點
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      if (board[y][x]) {
        lc.setLed(0, y, x, true);
      }
    }
  }

  // 畫正在掉落的點
  lc.setLed(0, fallingY, fallingX, true);
}

void spawnNewDot() {
  fallingX = random(0, WIDTH);
  fallingY = 0;
}

bool canMoveDown() {
  int nextY = fallingY + 1;

  // 到底了
  if (nextY >= HEIGHT) {
    return false;
  }

  // 下面已經有點了
  if (board[nextY][fallingX]) {
    return false;
  }

  return true;
}

void settleDot() {
  board[fallingY][fallingX] = true;
}

void setup() {
  lc.shutdown(0, false);   // 啟動 MAX7219
  lc.setIntensity(0, 6);   // 亮度 0~15
  lc.clearDisplay(0);

  randomSeed(analogRead(A0));

  clearBoard();
  spawnNewDot();
}

void loop() {
  drawBoard();
  delay(120);

  if (canMoveDown()) {
    fallingY++;
  } else {
    settleDot();

    if (isBoardFull()) {
      delay(500);

      // 全亮閃一下
      for (int y = 0; y < HEIGHT; y++) {
        lc.setRow(0, y, B11111111);
      }

      delay(500);

      lc.clearDisplay(0);
      clearBoard();
      delay(300);
    }

    spawnNewDot();

    // 如果新生成的位置已經被堵住，也清空重來
    if (board[fallingY][fallingX]) {
      delay(300);
      clearBoard();
    }
  }
}