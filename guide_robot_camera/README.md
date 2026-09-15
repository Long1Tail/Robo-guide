### Что это такое?

Пакет, по триггеру в топик  `/camera/trigger` публикует изображение в топик `/camera/image`

### Как использовать?

1. Пересобираем проект полностью
```
colcon build
source install/setup.bash
```
либо, если менялся только guide_robot_camera
```
colcon build --packages-select guide_robot_camera
source install
```

2. 
В терминале 1:
```
ros2 run guide_robot_camera camera_node   --ros-args   -p device:=/dev/usb_cam -p width:=640 -p height:=480 -p fps:=30   -p scale:=0.5 -p mono:=false   -p publish_raw:=true -p publish_compressed:=true   -p raw_every_n:=3 -p jpeg_quality:=80 -p target_fps:=30.0
```

В терминале 2:
```
ros2 launch guide_robot_camera camera.launch.py
```

По желанию, можно в терминале 3 запустить
```
ros2 run rqt_image_view rqt_image_view
```

И в терминал 4 закинуть

```
ros2 topic pub /camera/trigger std_msgs/msg/Bool "{data: true}" --once
```

### Какие проблемы?

Пока я делал этот пакет, он потерял свою актуальность (по моему субъективному мнению). 
Запуск
```
ros2 launch guide_robot_camera camera.launch.py
```
и так вызывает публикацию данных с камеры в топик /camera/image_raw.

###  Предложения/мысли в слух для улучшения
- Разобраться, как включать камеру только по триггеру и сразу выключать.
- добавить функцию обработки изображения в пакет. Камера будет передавать "сырые" фото, по триггеру получаем данные после какой-то обработки