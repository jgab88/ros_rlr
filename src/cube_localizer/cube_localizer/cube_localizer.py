#include <micro_ros_platformio.h>
#include <Wire.h>
#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <rmw_microros/rmw_microros.h>
#include <sensor_msgs/msg/imu.h>
#include <sensor_msgs/msg/magnetic_field.h>
#include <nav_msgs/msg/odometry.h>
#include <Adafruit_BNO055.h>
#include <SPI.h>
#include <Arduino.h>
#include <string.h>

#define LED_PIN 13
#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}
#define EXECUTE_EVERY_N_MS(MS, X) do { \
  static volatile int64_t init = -1; \
  if (init == -1) { init = uxr_millis();} \
  if (uxr_millis() - init > MS) { X; init = uxr_millis();} \
} while (0)

// Error indicator function
void error_loop() {
  while(1) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    delay(100);
  }
}

// Encoder pins and parameters
const int encoderPinA = 5;
const int encoderPinB = 6;
volatile long encoderCount = 0;
const float TICKS_PER_REVOLUTION = 1024.0;
const float WHEEL_DIAMETER = 0.1;
const float METERS_PER_TICK = (PI * WHEEL_DIAMETER) / TICKS_PER_REVOLUTION;
unsigned long previousTime = 0;
long previousCount = 0;

// Debug flag
bool debug_output = true;

// ROS entities
rclc_support_t support;
rcl_node_t node;
rcl_timer_t timer;
rclc_executor_t executor;
rcl_allocator_t allocator;
rcl_publisher_t imu_publisher;
rcl_publisher_t mag_publisher;
rcl_publisher_t odom_publisher;
sensor_msgs__msg__Imu imu_msg;
sensor_msgs__msg__MagneticField mag_msg;
nav_msgs__msg__Odometry odom_msg;
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28, &Wire);

enum states {
  WAITING_AGENT,
  AGENT_AVAILABLE,
  AGENT_CONNECTED,
  AGENT_DISCONNECTED
} state;

// Debug print function
void debug_print(const char* msg) {
  if (debug_output) {
    Serial.println(msg);
    Serial.flush(); // Ensure message is sent
  }
}

void FASTRUN encoderISR() {
    int a = digitalRead(encoderPinA);
    int b = digitalRead(encoderPinB);
    
    if (a == b) {
        encoderCount++;
    } else {
        encoderCount--;
    }
}

void timer_callback(rcl_timer_t * timer, int64_t last_call_time) {
  static unsigned long last_debug = 0;
  RCLC_UNUSED(last_call_time);
  
  if (timer != NULL) {
    unsigned long currentTime = micros();
    
    // Get IMU data with error checking
    imu::Quaternion quat = bno.getQuat();
    if (quat.w() == 0 && quat.x() == 0 && quat.y() == 0 && quat.z() == 0) {
      debug_print("WARNING: Invalid quaternion data");
      return;  // Skip this cycle if data is invalid
    }
    
    // Update message data
    imu_msg.orientation.x = quat.x();
    imu_msg.orientation.y = quat.y();
    imu_msg.orientation.z = quat.z();
    imu_msg.orientation.w = quat.w();

    // ... [rest of the sensor data collection] ...

    // Get timestamps
    currentTime = micros();  // Fresh timestamp
    uint32_t sec = currentTime / 1000000;
    uint32_t nanosec = (currentTime % 1000000) * 1000;

    // Update all headers
    imu_msg.header.stamp.sec = sec;
    imu_msg.header.stamp.nanosec = nanosec;
    mag_msg.header.stamp.sec = sec;
    mag_msg.header.stamp.nanosec = nanosec;
    odom_msg.header.stamp.sec = sec;
    odom_msg.header.stamp.nanosec = nanosec;

    // Set frame IDs
    static char imu_frame_id[] = "imu_link";
    static char odom_frame_id[] = "odom";
    static char base_frame_id[] = "base_link";
    
    imu_msg.header.frame_id.data = imu_frame_id;
    imu_msg.header.frame_id.size = strlen(imu_frame_id);
    mag_msg.header.frame_id.data = imu_frame_id;
    mag_msg.header.frame_id.size = strlen(imu_frame_id);
    odom_msg.header.frame_id.data = odom_frame_id;
    odom_msg.header.frame_id.size = strlen(odom_frame_id);
    odom_msg.child_frame_id.data = base_frame_id;
    odom_msg.child_frame_id.size = strlen(base_frame_id);

    // Publish with more robust error handling
    rcl_ret_t rc = rcl_publish(&imu_publisher, &imu_msg, NULL);
    if (rc != RCL_RET_OK) {
      debug_print("Failed to publish IMU");
      return;
    }

    rc = rcl_publish(&mag_publisher, &mag_msg, NULL);
    if (rc != RCL_RET_OK) {
      debug_print("Failed to publish MAG");
      return;
    }

    rc = rcl_publish(&odom_publisher, &odom_msg, NULL);
    if (rc != RCL_RET_OK) {
      debug_print("Failed to publish ODOM");
      return;
    }

    // Periodic debug output (every 1 second)
    if (currentTime - last_debug > 1000000) {
      debug_print("Publishing data successfully");
      last_debug = currentTime;
    }
  }
}

bool create_entities()
{
  debug_print("Creating entities...");
  
  allocator = rcl_get_default_allocator();
  
  // Initialize support with larger timeout
  rcl_ret_t rc = rclc_support_init(&support, 0, NULL, &allocator);
  if (rc != RCL_RET_OK) {
    debug_print("Failed to initialize support");
    return false;
  }

  // Create node with descriptive name
  rc = rclc_node_init_default(&node, "teensy_sensor_fusion", "", &support);
  if (rc != RCL_RET_OK) {
    debug_print("Failed to create node");
    return false;
  }

  // Create publishers
  rc = rclc_publisher_init_default(&imu_publisher, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, Imu), "imu/data");
  if (rc != RCL_RET_OK) {
    debug_print("Failed to create IMU publisher");
    return false;
  }

  // ... [Similar error checking for other publishers] ...

  // Create timer with longer timeout
  const unsigned int timer_timeout = 200;  // Increased from 100
  rc = rclc_timer_init_default(&timer, &support, RCL_MS_TO_NS(timer_timeout), timer_callback);
  if (rc != RCL_RET_OK) {
    debug_print("Failed to create timer");
    return false;
  }

  // Initialize executor
  executor = rclc_executor_get_zero_initialized_executor();
  rc = rclc_executor_init(&executor, &support.context, 1, &allocator);
  if (rc != RCL_RET_OK) {
    debug_print("Failed to initialize executor");
    return false;
  }

  rc = rclc_executor_add_timer(&executor, &timer);
  if (rc != RCL_RET_OK) {
    debug_print("Failed to add timer to executor");
    return false;
  }

  debug_print("Entities created successfully");
  return true;
}

void destroy_entities()
{
  debug_print("Destroying entities...");
  
  rmw_context_t * rmw_context = rcl_context_get_rmw_context(&support.context);
  (void) rmw_uros_set_context_entity_destroy_session_timeout(rmw_context, 0);

  rcl_publisher_fini(&imu_publisher, &node);
  rcl_publisher_fini(&mag_publisher, &node);
  rcl_publisher_fini(&odom_publisher, &node);
  rcl_timer_fini(&timer);
  rclc_executor_fini(&executor);
  rcl_node_fini(&node);
  rclc_support_fini(&support);

  debug_print("Entities destroyed");
}

void setup()
{
  // Initialize serial with higher baud rate and wait for it to be ready
  Serial.begin(115200);
  while (!Serial) {
    delay(100);
  }
  
  debug_print("\n\nStarting Teensy Sensor Fusion Node...");
  
  pinMode(LED_PIN, OUTPUT);
  
  // Initialize encoder pins
  pinMode(encoderPinA, INPUT_PULLUP);
  pinMode(encoderPinB, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(encoderPinA), encoderISR, CHANGE);
  
  // Configure micro-ROS
  set_microros_serial_transports(Serial);
  
  debug_print("Initializing BNO055...");
  if (!bno.begin()) {
    debug_print("Failed to initialize BNO055!");
    error_loop();
  }
  debug_print("BNO055 initialized");

  state = WAITING_AGENT;
}

void loop()
{
  static states previous_state = WAITING_AGENT;
  static unsigned long last_agent_check = 0;
  unsigned long current_time = millis();
  
  switch (state) {
    case WAITING_AGENT:
      if (previous_state != WAITING_AGENT) {
        debug_print("State: WAITING_AGENT");
        previous_state = WAITING_AGENT;
      }
      
      if (current_time - last_agent_check > 500) {
        state = (RMW_RET_OK == rmw_uros_ping_agent(500, 1)) ? AGENT_AVAILABLE : WAITING_AGENT;
        last_agent_check = current_time;
      }
      break;
      
    case AGENT_AVAILABLE:
      if (previous_state != AGENT_AVAILABLE) {
        debug_print("State: AGENT_AVAILABLE");
        previous_state = AGENT_AVAILABLE;
      }
      
      state = create_entities() ? AGENT_CONNECTED : WAITING_AGENT;
      if (state == WAITING_AGENT) {
        debug_print("Failed to create entities");
        destroy_entities();
      }
      break;
      
    case AGENT_CONNECTED:
      if (previous_state != AGENT_CONNECTED) {
        debug_print("State: AGENT_CONNECTED");
        previous_state = AGENT_CONNECTED;
      }
      
      if (current_time - last_agent_check > 200) {
        state = (RMW_RET_OK == rmw_uros_ping_agent(100, 1)) ? AGENT_CONNECTED : AGENT_DISCONNECTED;
        if (state == AGENT_DISCONNECTED) {
          debug_print("Agent ping failed");
        }
        last_agent_check = current_time;
      }
      
      if (state == AGENT_CONNECTED) {
        rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100));
      }
      break;
      
    case AGENT_DISCONNECTED:
      if (previous_state != AGENT_DISCONNECTED) {
        debug_print("State: AGENT_DISCONNECTED");
        previous_state = AGENT_DISCONNECTED;
      }
      destroy_entities();
      state = WAITING_AGENT;
      break;
      
    default:
      debug_print("ERROR: Invalid state!");
      break;
  }

  digitalWrite(LED_PIN, state == AGENT_CONNECTED);
}