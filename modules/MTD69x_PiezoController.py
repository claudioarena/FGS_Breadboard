import serial
import sys
import glob
import time

# Should work in both compatible or non compatible mode, but compatible mode should be faster


## Not implemented:

# DACSTEP?	Gets DAC step size used with up/down arrow keys. (1-5000).
# DACSTEP=	Sets DAC step size used with up/down arrow keys. (1-5000).
# Up Arrow	Increase selected channel by the set step size.
# Down Arrow	Decrease selected channel by the set step size.
# Right Arrow	Select next channel.
# Left Arrow	Select previous channel.

##Also, Echo on isn't implemented.
## Implementing it implies change to the code, to parse the echo response

class controller:

    ser = serial.Serial()
    commands = ""
    model = ""
    firmaware_version = ""
    voltage_range = ""
    serial_number = ""
    name = ""
    compatible = 0
    error_character = ''
    master_scan_enabled = 0
    voltage_commands_set = ["", "", "", ""]
    voltage_commands_get = ["", "", ""]
    voltage_max_commands_set = ["", "", "", ""]
    voltage_min_commands_set = ["", "", "", ""]
    voltage_max_commands_get = ["", "", "", ""]
    voltage_min_commands_get = ["", "", "", ""]

    def __init__(self, port=''):
        self.ser.baudrate = 115200
        self.ser.bytesize = serial.EIGHTBITS
        self.ser.parity = serial.PARITY_NONE
        self.ser.stopbits = serial.STOPBITS_ONE
        self.ser.xonxoff = False
        self.ser.timeout = 1
        self.ser.write_timeout = 1
        self.ser.setDTR(False)
        self.ser.setRTS(False)

        if port is not "":
            self.ser.port = port
            self.ser.open()
            self.ser.flushInput()
            self.ser.write('\n\r'.encode('utf-8'))
            res = self.ser.read(100)
            self.ser.close()
            if res != b'\n\rCMD_NOT_DEFINED>' and res != b'!':
                print("Wrong serial port!")
            else:
                self.ser.open()

        else:
            port = self.find_port() # This also opens the port

        self.set_echo_off()
        self.set_compatibility_on()

    def find_port(self):
        # Finds the serial port names
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i+1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.usbserial*')
        else:
            raise EnvironmentError('Error finding ports on your operating system')
        controller_port = ''
        for port in ports:
            try:
                self.ser.port = port
                self.ser.open()
                self.ser.flushInput()
                self.ser.write('\n\r'.encode('utf-8'))
                res = self.ser.read(100)
                self.ser.close()

                if res == b'!' or res == b'\n\rCMD_NOT_DEFINED>':
                    controller_port = port
                    break
            except (OSError, serial.SerialException):
                pass

        if controller_port == '':
            raise OSError('Cannot find MTD69x port')
        else:
            self.ser.port = controller_port
            self.ser.open()
            return controller_port

    def close(self):
        self.ser.close()

    def response_to_float(self, response):
        response = response.replace(']', '').replace('[', '').replace('*', '')
        return float(response)

    def send_query(self, query):
        self.ser.flushInput()
        self.ser.write((query + '\n\r').encode('utf-8'))
        if self.compatible:
            result = self.ser.read_until(b'*').decode('ascii')
        else:
            result = self.ser.read_until(b'>').decode('ascii')

        if result == '' or result == self.error_character:
            return 0

        return result[:-1]

    def send_command(self, command, value):
        self.ser.flushInput()
        self.ser.write((command + str(value) + '\n\r').encode('utf-8'))
        if self.compatible:
            result = self.ser.read_until(b'*').decode('ascii')
            if result == '*' or result.startswith('[') or "RESTORE" in result:
                return 1
            else:
                return 0
        else:
            result = self.ser.read_until(b'>').decode('ascii')
            if result == '>' or result.startswith('['):
                return 1
            else:
                return 0

    def get_available(self):
        result = self.send_query('?')
        result = result.replace('\r', '\r\n')
        print(result)
        self.commands = result.split('\r\n')

    def restore(self):
        return self.send_command('RESTORE', '')
        # return self.send_command('E=', 0) %Alternative command

    def set_echo_off(self):
        if self.compatible:
            return self.send_command('E', 0)
        else:
            return self.send_command('ECHO=', 0)
        # return self.send_command('E=', 0) %Alternative command

## NOT IMPLEMENTED.
## Implementing it implies change to the code, to parse the echo response
    # def set_echo_on(self):
    #     if self.compatible:
    #         self.send_command('E', 1)
    #     else:
    #         self.send_command('ECHO=', 1)
    #     # return self.send_command('E=', 1) %Alternative command

    def get_echo_status(self):
        result = self.send_query('ECHO?')
        if "Echo Off" in result:
            return 0
        elif "Echo On" in result:
            return 1
        else:
            return -1

    def get_id(self):
        if self.compatible:
            result = self.send_query('I')
        else:
            result = self.send_query('ID?')

        result = (result[3:-3]).replace('\r', '\r\n')
        print(result)
        result = result.split('\r\n')
        self.model = result[0]
        self.firmaware_version = result[1].split(': ')[1]
        self.voltage_range = result[2].split(': ')[1]
        self.serial_number = result[3].split(':')[1]
        self.name = result[4].split(':')[1]

    # def get_firmware(self):
    #    return self.send_query('I?')

    def get_name(self):
        name = self.send_query('FRIENDLY?')
        self.name = name.replace('\r>', '')
        return self.name

    def set_name(self, name):
        return self.send_command('FRIENDLY=', name)

    def get_serial_number(self):
        result = self.send_query('SERIAL?')
        self.serial_number = result[:-2]
        return self.serial_number

    def get_compatibility(self):
        result = self.send_query('cm?')[:-1]
        if "Compatibility Mode On" in result:
            return 1
        elif "Compatibility Mode Off" in result:
            return 0
        else:
            return -1

    def get_switch_limit(self):
        if self.compatible:
            result = self.send_query('%')[:-1]
        else:
            result = self.send_query('VLIMIT?')[:-1]

        return self.response_to_float(result)

    def get_rotary_mode(self):
        result = self.send_query('ROTARYMODE?')[:-1]
        return int(result)

    def set_rotary_mode(self, mode):
        return self.send_command("ROTARYMODE=", mode)

    def enable_push_mode(self):
        result = self.send_command('PUSHDISABLE=', 0)
        return result

    def disable_push_mode(self):
        result = self.send_command('PUSHDISABLE=', 1)
        return result

    def get_push_mode(self):
        result = self.send_query('PUSHDISABLE?')[:-1]
        return int(result)

    def enable_master_scan(self):
        result = self.send_command('MSENABLE=', 1)
        self.master_scan_enabled = 1
        return result

    def disable_master_scan(self):
        result = self.send_command('MSENABLE=', 0)
        master_scan_enabled = 0
        return result

    def get_master_scan_state(self):
        result = self.send_query('MSENABLE?')[:-1]
        master_scan_enabled = int(result)
        return int(result)

    def set_master_voltage(self, voltage):
        self.get_master_scan_state()
        if self.master_scan_enabled:
            result = self.send_command('MSVOLTAGE=', voltage)
        else:
            print("Master Scan not enabled")
            return 0

        return result

    def get_master_voltage(self):
        result = self.send_query('MSVOLTAGE?')
        result = self.response_to_float(result)
        return result

    def set_intensity(self, intensity):
        if intensity < 0:
            intensity = 0
        elif intensity > 15:
            intensity = 15

        result = self.send_command('INTENSITY=', intensity)
        return result

    def get_intensity(self):
        result = self.send_query('INTENSITY?')[:-1]
        return int(result)

    def set_compatibility_on(self):
        return_value = self.send_command('cm=', 1)
        self.compatible = 1
        self.error_character = '*'
        self.voltage_commands_set = ["XV", "XY", "XZ", "AV"]
        self.voltage_commands_get = ["XR", "YR", "ZR"]
        self.voltage_max_commands_set = ["XH", "YH", "ZH", "SYSMAX="]
        self.voltage_min_commands_set = ["XL", "YL", "ZL", "SYSMIN="]
        self.voltage_max_commands_get = ["XH?", "YH?", "ZH?", "SYSMAX?"]
        self.voltage_min_commands_get = ["XL?", "YL?", "ZL?", "SYSMIN?"]
        return return_value

    def set_compatibility_off(self):
        return_value = self.send_command('cm=', 0)
        self.error_character = 'CMD_NOT_DEFINED>'
        self.voltage_commands_set = ["XVOLTAGE=", "YVOLTAGE=", "ZVOLTAGE=", "ALLVOLTAGE="]
        self.voltage_commands_get = ["XVOLTAGE?", "YVOLTAGE?", "ZVOLTAGE?"]
        self.voltage_max_commands_set = ["XMAX=", "YMAX=", "ZMAX=", "SYSMAX="]
        self.voltage_min_commands_set = ["XMIN=", "YMIN=", "ZMIN=", "SYSMIN="]
        self.voltage_max_commands_get = ["XMAX?", "YMAX?", "ZMAX?", "SYSMAX?"]
        self.voltage_min_commands_get = ["XMIN?", "YMIN?", "ZMIN?", "SYSMIN?"]
        self.compatible = 0
        return return_value

    def get_x_voltage(self):
        value = self.send_query(self.voltage_commands_get[0])
        return self.response_to_float(value)

    def get_y_voltage(self):
        value = self.send_query(self.voltage_commands_get[1])
        return self.response_to_float(value)

    def get_z_voltage(self):
        value = self.send_query(self.voltage_commands_get[2])
        return self.response_to_float(value)

    def set_x_voltage(self, voltage):
        return self.send_command(self.voltage_commands_set[0], voltage)

    def set_y_voltage(self, voltage):
        return self.send_command(self.voltage_commands_set[1], voltage)

    def set_z_voltage(self, voltage):
        return self.send_command(self.voltage_commands_set[2], voltage)

    def set_all_voltage(self, voltage):
        return self.send_command(self.voltage_commands_set[3], voltage)

    def set_xyz_voltage(self, voltage_x, voltage_y, voltage_z):
        return self.send_command("XYZVOLTAGE=", str("%.4f,%.4f,%.4f" % (voltage_x, voltage_y, voltage_z)))

    def get_xyz_voltage(self):
        result = self.send_query("XYZVOLTAGE?")
        result = result.replace("[ ", "").replace("]", "").replace("\r", "").replace(" ", "")
        result = result.split(",")
        result = [float(i) for i in result]
        return result

    def get_x_voltage_max(self):
        value = self.send_query(self.voltage_max_commands_get[0])
        return self.response_to_float(value)

    def get_x_voltage_min(self):
        value = self.send_query(self.voltage_min_commands_get[0])
        return self.response_to_float(value)

    def get_y_voltage_max(self):
        value = self.send_query(self.voltage_max_commands_get[1])
        return self.response_to_float(value)

    def get_y_voltage_min(self):
        value = self.send_query(self.voltage_min_commands_get[1])
        return self.response_to_float(value)

    def get_z_voltage_max(self):
        value = self.send_query(self.voltage_max_commands_get[2])
        return self.response_to_float(value)

    def get_z_voltage_min(self):
        value = self.send_query(self.voltage_min_commands_get[2])
        return self.response_to_float(value)

    def get_sys_voltage_max(self):
        value = self.send_query(self.voltage_max_commands_get[3])
        return self.response_to_float(value)

    def get_sys_voltage_min(self):
        value = self.send_query(self.voltage_min_commands_get[2])
        return self.response_to_float(value)

    def set_x_voltage_max(self, voltage):
        return self.send_command(self.voltage_max_commands_set[0], voltage)

    def set_x_voltage_min(self, voltage):
        return self.send_command(self.voltage_min_commands_set[0], voltage)

    def set_y_voltage_max(self, voltage):
        return self.send_command(self.voltage_max_commands_set[1], voltage)

    def set_y_voltage_min(self, voltage):
        return self.send_command(self.voltage_min_commands_set[1], voltage)

    def set_z_voltage_max(self, voltage):
        return self.send_command(self.voltage_max_commands_set[2], voltage)

    def set_z_voltage_min(self, voltage):
        return self.send_command(self.voltage_min_commands_set[2], voltage)

    def set_sys_voltage_max(self, voltage):
        return self.send_command(self.voltage_max_commands_set[3], voltage)

    def set_z_voltage_min(self, voltage):
        return self.send_command(self.voltage_min_commands_set[3], voltage)


if __name__ == '__main__':
    print("MTD69x Piezo controller")