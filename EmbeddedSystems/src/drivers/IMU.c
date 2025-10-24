#include "IMU.h"
#include "driver/i2c.h"

#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_mac.h" // optional if you use esp_mac functions

// #define portTICK_PERIOD_MS   ( (TickType_t) 1000 / configTICK_RATE_HZ )
// #include "portmacro.h"

static const char *TAG = "MPU6050";

// Offsets
static float accel_offset_x = 0, accel_offset_y = 0, accel_offset_z = 0;
static float gyro_offset_x = 0, gyro_offset_y = 0, gyro_offset_z = 0;

static esp_err_t i2c_write_byte(uint8_t reg, uint8_t data)
{
    return i2c_master_write_to_device(I2C_NUM_0, MPU6050_ADDR,
                                      (uint8_t[]){reg, data}, 2, 1000 / portTICK_PERIOD_MS);
}

static esp_err_t i2c_read_bytes(uint8_t reg, uint8_t *data, size_t len)
{
    return i2c_master_write_read_device(I2C_NUM_0, MPU6050_ADDR,
                                        &reg, 1, data, len, 1000 / portTICK_PERIOD_MS);
}

void imu_init(void)
{
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ};
    i2c_param_config(I2C_NUM_0, &conf);
    i2c_driver_install(I2C_NUM_0, conf.mode, 0, 0, 0);

    i2c_write_byte(MPU6050_REG_PWR_MGMT_1, 0x00);   // Wake up
    i2c_write_byte(MPU6050_REG_SMPLRT_DIV, 0x07);   // 1 kHz / (1 + 7) = 125 Hz
    i2c_write_byte(MPU6050_REG_CONFIG, 0x00);       // No DLPF
    i2c_write_byte(MPU6050_REG_GYRO_CONFIG, 0x00);  // ±250 °/s
    i2c_write_byte(MPU6050_REG_ACCEL_CONFIG, 0x00); // ±2g
    ESP_LOGI(TAG, "IMU initialized");
}

void imu_set_accel_range(uint8_t range)
{
    i2c_write_byte(MPU6050_REG_ACCEL_CONFIG, range << 3);
}

void imu_set_gyro_range(uint8_t range)
{
    i2c_write_byte(MPU6050_REG_GYRO_CONFIG, range << 3);
}

void imu_read_accel(int16_t *ax, int16_t *ay, int16_t *az)
{
    uint8_t data[6];
    i2c_read_bytes(MPU6050_REG_ACCEL_XOUT_H, data, 6);
    *ax = (data[0] << 8) | data[1];
    *ay = (data[2] << 8) | data[3];
    *az = (data[4] << 8) | data[5];
}

void imu_read_gyro(int16_t *gx, int16_t *gy, int16_t *gz)
{
    uint8_t data[6];
    i2c_read_bytes(0x43, data, 6);
    *gx = (data[0] << 8) | data[1];
    *gy = (data[2] << 8) | data[3];
    *gz = (data[4] << 8) | data[5];
}

void imu_read_all(int16_t *ax, int16_t *ay, int16_t *az,
                  int16_t *gx, int16_t *gy, int16_t *gz)
{
    uint8_t data[14];
    i2c_read_bytes(MPU6050_REG_ACCEL_XOUT_H, data, 14);
    *ax = (data[0] << 8) | data[1];
    *ay = (data[2] << 8) | data[3];
    *az = (data[4] << 8) | data[5];
    *gx = (data[8] << 8) | data[9];
    *gy = (data[10] << 8) | data[11];
    *gz = (data[12] << 8) | data[13];
}

float imu_convert_accel(int16_t raw_value)
{
    return (float)raw_value / 16384.0f; // ±2g
}

float imu_convert_gyro(int16_t raw_value)
{
    return (float)raw_value / 131.0f; // ±250°/s
}

void imu_calibrate(void)
{
    int16_t ax, ay, az, gx, gy, gz;
    long a_x = 0, a_y = 0, a_z = 0, g_x = 0, g_y = 0, g_z = 0;

    for (int i = 0; i < 200; i++)
    {
        imu_read_all(&ax, &ay, &az, &gx, &gy, &gz);
        a_x += ax;
        a_y += ay;
        a_z += az;
        g_x += gx;
        g_y += gy;
        g_z += gz;
        vTaskDelay(10 / portTICK_PERIOD_MS);
    }
    accel_offset_x = a_x / 200.0f;
    accel_offset_y = a_y / 200.0f;
    accel_offset_z = a_z / 200.0f;
    gyro_offset_x = g_x / 200.0f;
    gyro_offset_y = g_y / 200.0f;
    gyro_offset_z = g_z / 200.0f;
    ESP_LOGI(TAG, "IMU calibrated");
}
