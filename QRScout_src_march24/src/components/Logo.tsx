
import logo from '../assets/1024-kil-a-bytes.png';

export function Logo() {
  return (
    <img
      src={logo}
      alt="QRScout Logo"
      style={{ height: 148, marginLeft:72, transform: 'translateY(-32px)' }}
    />
  );
}

