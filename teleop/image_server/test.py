#!/usr/bin/env python3
"""
Intel RealSense 设备序列号检查脚本
检查所有连接的RealSense设备并显示详细信息
"""

import pyrealsense2 as rs
import json
from datetime import datetime


def get_device_info(device):
    """获取设备详细信息"""
    info = {}
    
    # 获取基本信息
    try:
        info['name'] = device.get_info(rs.camera_info.name)
    except:
        info['name'] = "Unknown"
    
    try:
        info['serial_number'] = device.get_info(rs.camera_info.serial_number)
    except:
        info['serial_number'] = "Unknown"
    
    try:
        info['firmware_version'] = device.get_info(rs.camera_info.firmware_version)
    except:
        info['firmware_version'] = "Unknown"
    
    try:
        info['product_id'] = device.get_info(rs.camera_info.product_id)
    except:
        info['product_id'] = "Unknown"
    
    try:
        info['product_line'] = device.get_info(rs.camera_info.product_line)
    except:
        info['product_line'] = "Unknown"
    
    try:
        info['physical_port'] = device.get_info(rs.camera_info.physical_port)
    except:
        info['physical_port'] = "Unknown"
    
    # 获取传感器信息
    sensors = []
    for sensor in device.query_sensors():
        sensor_info = {}
        try:
            sensor_info['name'] = sensor.get_info(rs.camera_info.name)
        except:
            sensor_info['name'] = "Unknown Sensor"
        
        # 获取支持的流
        profiles = []
        try:
            for profile in sensor.get_stream_profiles():
                if profile.is_video_stream_profile():
                    vp = profile.as_video_stream_profile()
                    profiles.append({
                        'stream_type': str(profile.stream_type()),
                        'format': str(profile.format()),
                        'width': vp.width(),
                        'height': vp.height(),
                        'fps': profile.fps()
                    })
                else:
                    profiles.append({
                        'stream_type': str(profile.stream_type()),
                        'format': str(profile.format()),
                        'fps': profile.fps()
                    })
        except Exception as e:
            profiles.append({'error': f"无法获取流信息: {str(e)}"})
        
        sensor_info['supported_profiles'] = profiles
        sensors.append(sensor_info)
    
    info['sensors'] = sensors
    return info


def check_realsense_devices():
    """检查所有RealSense设备"""
    print("=" * 60)
    print("Intel RealSense 设备检查")
    print("=" * 60)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 创建上下文
        ctx = rs.context()
        
        # 获取所有设备
        devices = ctx.query_devices()
        device_count = len(devices)
        
        print(f"发现 {device_count} 个RealSense设备")
        print()
        
        if device_count == 0:
            print("❌ 没有发现任何RealSense设备")
            print("请检查:")
            print("  - 设备是否已连接")
            print("  - USB驱动是否正确安装")
            print("  - 设备是否被其他程序占用")
            return []
        
        devices_info = []
        
        for i, device in enumerate(devices):
            print(f"设备 {i+1}:")
            print("-" * 40)
            
            try:
                device_info = get_device_info(device)
                devices_info.append(device_info)
                
                # 打印基本信息
                print(f"📷 设备名称: {device_info['name']}")
                print(f"🔢 序列号: {device_info['serial_number']}")
                print(f"🔧 固件版本: {device_info['firmware_version']}")
                print(f"🆔 产品ID: {device_info['product_id']}")
                print(f"📦 产品线: {device_info['product_line']}")
                print(f"🔌 物理端口: {device_info['physical_port']}")
                
                # 打印传感器信息
                print(f"📡 传感器数量: {len(device_info['sensors'])}")
                for j, sensor in enumerate(device_info['sensors']):
                    print(f"   传感器 {j+1}: {sensor['name']}")
                    if 'supported_profiles' in sensor:
                        unique_streams = set()
                        for profile in sensor['supported_profiles']:
                            if 'stream_type' in profile:
                                unique_streams.add(profile['stream_type'])
                        if unique_streams:
                            print(f"      支持的流: {', '.join(unique_streams)}")
                
            except Exception as e:
                print(f"❌ 获取设备信息失败: {str(e)}")
                devices_info.append({'error': str(e), 'device_index': i})
            
            print()
        
        return devices_info
        
    except Exception as e:
        print(f"❌ 初始化RealSense失败: {str(e)}")
        print("请确保:")
        print("  - 已安装pyrealsense2库: pip install pyrealsense2")
        print("  - RealSense SDK已正确安装")
        return []


def save_device_info(devices_info, filename="realsense_devices.json"):
    """保存设备信息到JSON文件"""
    try:
        output_data = {
            'scan_time': datetime.now().isoformat(),
            'device_count': len(devices_info),
            'devices': devices_info
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 设备信息已保存到: {filename}")
        
    except Exception as e:
        print(f"❌ 保存文件失败: {str(e)}")


def print_summary(devices_info):
    """打印摘要信息"""
    if not devices_info:
        return
    
    print("=" * 60)
    print("设备摘要")
    print("=" * 60)
    
    for i, device in enumerate(devices_info):
        if 'error' not in device:
            print(f"{i+1}. {device.get('name', 'Unknown')} - SN: {device.get('serial_number', 'Unknown')}")
        else:
            print(f"{i+1}. 设备错误: {device['error']}")
    
    print()


if __name__ == "__main__":
    try:
        # 检查设备
        devices_info = check_realsense_devices()
        
        # 打印摘要
        print_summary(devices_info)
        
        # 保存到文件
        if devices_info:
            save_to_file = input("是否保存设备信息到JSON文件? (y/n): ").lower().strip()
            if save_to_file in ['y', 'yes', '是']:
                filename = input("输入文件名 (按回车使用默认名 'realsense_devices.json'): ").strip()
                if not filename:
                    filename = "realsense_devices.json"
                save_device_info(devices_info, filename)
        
        # 持续监控选项
        monitor = input("是否启用持续监控模式? (y/n): ").lower().strip()
        if monitor in ['y', 'yes', '是']:
            import time
            print("持续监控模式已启用，按 Ctrl+C 退出...")
            try:
                while True:
                    time.sleep(5)
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 重新扫描设备...")
                    new_devices = check_realsense_devices()
                    print_summary(new_devices)
            except KeyboardInterrupt:
                print("\n监控已停止")
        
    except KeyboardInterrupt:
        print("\n程序已退出")
    except Exception as e:
        print(f"\n程序出错: {str(e)}")




# python3 -c "
# import pyrealsense2 as rs
# ctx = rs.context()
# devices = ctx.query_devices()

# for dev in devices:
#     print(f'Device: {dev.get_info(rs.camera_info.serial_number)}')
#     for sensor in dev.query_sensors():
#         print(f'  Sensor: {sensor.get_info(rs.camera_info.name)}')
#         profiles = sensor.get_stream_profiles()
#         for profile in profiles:
#             if profile.stream_type() == rs.stream.color:
#                 video_profile = profile.as_video_stream_profile()
#                 print(f'    Color: {video_profile.width()}x{video_profile.height()} @ {video_profile.fps()}fps')
# "