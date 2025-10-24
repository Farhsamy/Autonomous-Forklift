

#ifndef IMU_H
#define IMU_H

#include <stdint.h>
#define portTICK_PERIOD_MS ( ( TickType_t ) 1000 / configTICK_RATE_HZ )
// I2C config
#define I2C_MASTER_SCL_IO  9      // SCL pin
#define I2C_MASTER_SDA_IO  8      // SDA pin
#define I2C_MASTER_FREQ_HZ 400000
#define MPU6050_ADDR       0x68

// MPU6050 registers
#define MPU6050_REG_PWR_MGMT_1  0x6B
#define MPU6050_REG_SMPLRT_DIV  0x19
#define MPU6050_REG_CONFIG      0x1A
#define MPU6050_REG_GYRO_CONFIG 0x1B
#define MPU6050_REG_ACCEL_CONFIG 0x1C
#define MPU6050_REG_ACCEL_XOUT_H 0x3B

// Function prototypes
void imu_init(void);                                             // Initializes I2C and configures MPU6050 registers.
void imu_read_accel(int16_t *ax, int16_t *ay, int16_t *az);      // Reads accelerometer data in raw integer form.
void imu_read_gyro(int16_t *gx, int16_t *gy, int16_t *gz);       // Reads gyroscope data in raw integer form.
void imu_read_all(int16_t *ax, int16_t *ay, int16_t *az,       
                  int16_t *gx, int16_t *gy, int16_t *gz);        // Reads all motion data at once (efficient for real-time control).
void imu_set_accel_range(uint8_t range);                         // Selects sensitivity range for accelerometer.
void imu_set_gyro_range(uint8_t range);                          // Selects sensitivity range for gyroscope.
float imu_convert_accel(int16_t raw_value);                      // Converts raw accel readings to g units.
float imu_convert_gyro(int16_t raw_value);                       // Converts raw gyro readings to degrees per second.
void imu_calibrate(void);                                        // Measures zero-offsets to improve precision.

#endif
