

#include "Encoder.h"

// Helper to encode pin states into a 2-bit value
static uint8_t encodeAB(bool pinA, bool pinB) {
    return (pinA << 1) | pinB;
}

void Encoder_Init(Encoder *enc, bool pinA, bool pinB) {
    enc->position = 0;
    enc->direction = ENCODER_DIR_NONE;
    enc->prev_state = encodeAB(pinA, pinB);
}

void Encoder_Update(Encoder *enc, bool pinA, bool pinB) {
    uint8_t curr_state = encodeAB(pinA, pinB);
    uint8_t transition = (enc->prev_state << 2) | curr_state;

    switch (transition) {
        case 0b0001:
        case 0b0111:
        case 0b1110:
        case 0b1000:
            enc->position++;
            enc->direction = ENCODER_DIR_CW;
            break;
        case 0b0010:
        case 0b0100:
        case 0b1101:
        case 0b1011:
            enc->position--;
            enc->direction = ENCODER_DIR_CCW;
            break;
        default:
            enc->direction = ENCODER_DIR_NONE;
            break;
    }

    enc->prev_state = curr_state;
}

int32_t Encoder_GetPosition(const Encoder *enc) {
    return enc->position;
}

EncoderDirection Encoder_GetDirection(const Encoder *enc) {
    return enc->direction;
}
