from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, SelectField, TextAreaField, BooleanField, HiddenField
from wtforms.validators import DataRequired, Length, Optional, EqualTo, Email, ValidationError
from wtforms.widgets import TextArea

# Все города России (крупные)
RUSSIAN_CITIES = [
    ('', 'Выберите город'),
    ('Москва', 'Москва'),
    ('Санкт-Петербург', 'Санкт-Петербург'),
    ('Новосибирск', 'Новосибирск'),
    ('Екатеринбург', 'Екатеринбург'),
    ('Казань', 'Казань'),
    ('Нижний Новгород', 'Нижний Новгород'),
    ('Челябинск', 'Челябинск'),
    ('Самара', 'Самара'),
    ('Омск', 'Омск'),
    ('Ростов-на-Дону', 'Ростов-на-Дону'),
    ('Уфа', 'Уфа'),
    ('Красноярск', 'Красноярск'),
    ('Пермь', 'Пермь'),
    ('Воронеж', 'Воронеж'),
    ('Волгоград', 'Волгоград'),
    ('Краснодар', 'Краснодар'),
    ('Саратов', 'Саратов'),
    ('Тюмень', 'Тюмень'),
    ('Тольятти', 'Тольятти'),
    ('Ижевск', 'Ижевск'),
    ('Барнаул', 'Барнаул'),
    ('Ульяновск', 'Ульяновск'),
    ('Иркутск', 'Иркутск'),
    ('Хабаровск', 'Хабаровск'),
    ('Ярославль', 'Ярославль'),
    ('Владивосток', 'Владивосток'),
    ('Махачкала', 'Махачкала'),
    ('Томск', 'Томск'),
    ('Оренбург', 'Оренбург'),
    ('Кемерово', 'Кемерово'),
    ('Новокузнецк', 'Новокузнецк'),
    ('Рязань', 'Рязань'),
    ('Астрахань', 'Астрахань'),
    ('Набережные Челны', 'Набережные Челны'),
    ('Пенза', 'Пенза'),
    ('Киров', 'Киров'),
    ('Липецк', 'Липецк'),
    ('Балашиха', 'Балашиха'),
    ('Чебоксары', 'Чебоксары'),
    ('Калининград', 'Калининград'),
    ('Тула', 'Тула'),
    ('Курск', 'Курск'),
    ('Ставрополь', 'Ставрополь'),
    ('Улан-Удэ', 'Улан-Удэ'),
    ('Тверь', 'Тверь'),
    ('Магнитогорск', 'Магнитогорск'),
    ('Севастополь', 'Севастополь'),
    ('Сочи', 'Сочи'),
    ('Белгород', 'Белгород'),
    ('Нижний Тагил', 'Нижний Тагил'),
    ('Владимир', 'Владимир'),
    ('Архангельск', 'Архангельск'),
    ('Симферополь', 'Симферополь'),
    ('Чита', 'Чита'),
    ('Мытищи', 'Мытищи'),
    ('Калуга', 'Калуга'),
    ('Вологда', 'Вологда'),
    ('Саранск', 'Саранск'),
    ('Смоленск', 'Смоленск'),
    ('Курган', 'Курган'),
    ('Брянск', 'Брянск'),
    ('Владикавказ', 'Владикавказ'),
    ('Грозный', 'Грозный'),
    ('Якутск', 'Якутск'),
    ('Петрозаводск', 'Петрозаводск'),
    ('Мурманск', 'Мурманск'),
    ('Нальчик', 'Нальчик'),
    ('Сыктывкар', 'Сыктывкар'),
    ('Йошкар-Ола', 'Йошкар-Ола'),
    ('Иваново', 'Иваново'),
    ('Кострома', 'Кострома'),
    ('Орёл', 'Орёл'),
    ('Тамбов', 'Тамбов'),
    ('Псков', 'Псков'),
    ('Новгород', 'Новгород'),
    ('Ханты-Мансийск', 'Ханты-Мансийск'),
    ('Салехард', 'Салехард'),
    ('Анадырь', 'Анадырь'),
    ('Петропавловск-Камчатский', 'Петропавловск-Камчатский'),
    ('Южно-Сахалинск', 'Южно-Сахалинск'),
    ('Магадан', 'Магадан'),
    ('Биробиджан', 'Биробиджан'),
    ('Горно-Алтайск', 'Горно-Алтайск'),
    ('Абакан', 'Абакан'),
    ('Кызыл', 'Кызыл'),
    ('Усть-Орда', 'Усть-Орда'),
    ('Агинское', 'Агинское'),
    ('Дудинка', 'Дудинка'),
    ('Норильск', 'Норильск'),
    ('Нефтеюганск', 'Нефтеюганск'),
    ('Новочебоксарск', 'Новочебоксарск'),
    ('Альметьевск', 'Альметьевск'),
    ('Братск', 'Братск'),
    ('Великий Новгород', 'Великий Новгород'),
    ('Дзержинск', 'Дзержинск'),
    ('Елец', 'Елец'),
    ('Жуковский', 'Жуковский'),
    ('Зеленоград', 'Зеленоград'),
    ('Коломна', 'Коломна'),
    ('Королёв', 'Королёв'),
    ('Люберцы', 'Люберцы'),
    ('Одинцово', 'Одинцово'),
    ('Подольск', 'Подольск'),
    ('Серпухов', 'Серпухов'),
    ('Химки', 'Химки'),
    ('Щёлково', 'Щёлково'),
]

class LoginForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired(), Length(min=3, max=80)])
    password = PasswordField('Пароль', validators=[DataRequired()])
    submit = SubmitField('Войти')

class RegisterForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired(), Length(min=3, max=80)])
    password = PasswordField('Пароль', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Подтвердите пароль', validators=[DataRequired(), EqualTo('password')])
    description = TextAreaField('О себе', validators=[Optional()])
    city = SelectField('Город', choices=RUSSIAN_CITIES, validators=[Optional()])
    gender = SelectField('Пол', choices=[
        ('', 'Не указано'),
        ('male', 'Мужской'),
        ('female', 'Женский'),
        ('other', 'Другой')
    ], validators=[Optional()])
    interests = TextAreaField('Интересы', validators=[Optional()])
    submit = SubmitField('Зарегистрироваться')

class StoryForm(FlaskForm):
    content = TextAreaField('Содержание', validators=[DataRequired()])
    image = FileField('Изображение')
    submit = SubmitField('Опубликовать историю')

class AddFriendForm(FlaskForm):
    submit = SubmitField('Добавить в друзья')

class SearchForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[Optional()])
    submit = SubmitField('Найти')

class PostForm(FlaskForm):
    content = TextAreaField('Содержание', validators=[DataRequired()])
    image = FileField('Изображение')
    submit = SubmitField('Опубликовать')

class CommentForm(FlaskForm):
    content = TextAreaField('Комментарий', validators=[DataRequired()])
    submit = SubmitField('Отправить')

class AnonymousChatForm(FlaskForm):
    room_id = HiddenField('ID комнаты', validators=[Optional()])
    content = TextAreaField('Сообщение', validators=[DataRequired()])
    submit = SubmitField('Отправить')

class ProfileForm(FlaskForm):
    description = TextAreaField('О себе', validators=[Optional()])
    relationship_status = SelectField('Статус отношений', choices=[
        ('не интересуюсь', 'Не интересуюсь'),
        ('в поиске', 'В поиске'),
        ('в отношениях', 'В отношениях'),
        ('помолвлен(а)', 'Помолвлен(а)'),
        ('женат/замужем', 'Женат/Замужем'),
        ('всё сложно', 'Всё сложно')
    ], validators=[Optional()])
    city = SelectField('Город', choices=RUSSIAN_CITIES, validators=[Optional()])
    gender = SelectField('Пол', choices=[
        ('', 'Не указано'),
        ('male', 'Мужской'),
        ('female', 'Женский'),
        ('other', 'Другой')
    ], validators=[Optional()])
    interests = TextAreaField('Интересы', validators=[Optional()])
    avatar = FileField('Аватар')
    submit = SubmitField('Сохранить')
