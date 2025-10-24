

#ifndef ENCODER_H
#define ENCODER_H

#include <stdint.h>
#include <stdbool.h>

// Encoder direction enum
typedef enum {
    ENCODER_DIR_NONE = 0,
    ENCODER_DIR_CW,
    ENCODER_DIR_CCW
} EncoderDirection;

// Encoder state structure
typedef struct {
    int32_t position;
    EncoderDirection direction;
    uint8_t prev_state;
} Encoder;


void Encoder_Init(Encoder *enc, bool pinA, bool pinB);        // Initialize encoder with initial pin states
void Encoder_Update(Encoder *enc, bool pinA, bool pinB);      // Update encoder state based on current pin values
int32_t Encoder_GetPosition(const Encoder *enc);              // Get current encoder position
EncoderDirection Encoder_GetDirection(const Encoder *enc);    // Get last detected direction

#endif // ENCODER_H
