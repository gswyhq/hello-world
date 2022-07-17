测试：
执行在项目目录项目运行：pytest

~\hello-world\fastapi\my_test $ pytest
============================= test session starts =============================
platform win32 -- Python 3.6.5, pytest-3.10.1, py-1.10.0, pluggy-0.13.1
rootdir: ~\hello-world\fastapi\my_test, inifile:
plugins: typeguard-2.12.1, arraydiff-0.2, doctestplus-0.1.3, openfiles-0.3.0, remotedata-0.2.1
collected 6 items

test_main_b.py ......                                                    [100%]

========================== 6 passed in 1.38 seconds ===========================



若报错：
AttributeError: 'Function' object has no attribute 'get_marker'
则降低pytest版本：
pip install pytest==3.10.1

