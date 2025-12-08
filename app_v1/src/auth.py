import bcrypt
from src.database_manager import DatabaseManager

class AuthManager:
    """Gestiona la autenticación y registro de usuarios."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def login(self, username, password):
        """Verifica credenciales."""
        user_data = self.db.get_user(username)
        if user_data:
            stored_hash = user_data[0]
            if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                return True
        return False

    def register(self, username, password):
        """Crea un nuevo usuario con contraseña hasheada."""
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        return self.db.add_user(username, hashed)