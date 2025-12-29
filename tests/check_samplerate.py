import sounddevice as sd

def check_samplerate(device_id):
    try:
        dev_info = sd.query_devices(device_id)
        print(f"\n--- Device {device_id} Info ---")
        print(f"Name: {dev_info['name']}")
        print(f"Default Sample Rate: {dev_info['default_samplerate']}")
        
        # Test common rates
        rates = [16000, 44100, 48000]
        print("native support check:")
        for r in rates:
            try:
                if dev_info['max_input_channels'] > 0:
                    sd.check_input_settings(device=device_id, channels=1, dtype='float32', extra_settings=None, samplerate=r)
                    print(f"  Input  {r}Hz: OK")
                if dev_info['max_output_channels'] > 0:
                    sd.check_output_settings(device=device_id, channels=1, dtype='float32', extra_settings=None, samplerate=r)
                    print(f"  Output {r}Hz: OK")
            except Exception as e:
                print(f"  {r}Hz: Failed ({e})")
                
    except Exception as e:
        print(f"Error checking device {device_id}: {e}")

if __name__ == "__main__":
    # Check Stereo Mix (Input) and Realtek Speakers (Output)
    check_samplerate(31) 
    check_samplerate(33)
