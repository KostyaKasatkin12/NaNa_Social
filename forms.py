from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, SelectField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired()])
    description = StringField('Description')
    city = StringField('City')
    gender = SelectField('Gender', choices=[('Male', 'Male'), ('Female', 'Female')], default='Male')
    interests = StringField('Interests')
    submit = SubmitField('Register')

class StoryForm(FlaskForm):
    content = StringField('Content', validators=[DataRequired()])
    image = FileField('Image', validators=[DataRequired()])
    submit = SubmitField('Add Story')

class AddFriendForm(FlaskForm):
    submit = SubmitField('Добавить в друзья')

class SearchForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    submit = SubmitField('Search')
