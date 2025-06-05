from .db import db, migrate
def init_db(app):
    db.init_app(app)
    migrate.init_app(app, db)