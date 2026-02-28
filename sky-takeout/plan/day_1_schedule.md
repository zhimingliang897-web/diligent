# Day 1: 苍穹外卖环境搭建与员工登录详细教程

## 🎯 学习目标
手把手带你完成 IntelliJ IDEA 的项目导入、数据库表的二次确认，并一步步敲击代码完成员工登录接口、JWT拦截器，以及员工的分页查询。

---

## 🛠 一、 数据库与工程基础准备

### 1. 数据库确认 (MySQL)
我们在 Day 0 已经建好了数据库，今天需要确认数据库连接可用。
1. 打开终端（CMD/PowerShell），输入：`mysql -u root -p`，输入密码登录。
2. 确认库表是否正常：
    ```sql
    USE sky_takeout;
    SHOW TABLES;
    SELECT * FROM employee;
    ```
    *(你应该能看到一条 admin 的默认员工记录，密码是 `13800000000` MD5加密后的值)*

### 2. IntelliJ IDEA 项目导入与结构
通常苍穹外卖提供了一个基础代码骨架，我们需要在 IDEA 里正确打开：
1. 打开 IDEA -> `File` -> `Open` -> 选择 `sky-takeout` 的根目录。
2. 确保它是一个 Maven 工程：右侧应该有 `Maven` 侧边栏。如果没有，右键根目录的 `pom.xml` -> `Add as Maven Project`。
3. 检查 JDK 版本：`File` -> `Project Structure` -> `Project`，设置 SDK 为 Java 8 或 11/17（按你本地安装的来）。
4. **修改数据库密码**: 打开 `sky-server/src/main/resources/application-dev.yml`：
    ```yaml
    spring:
      datasource:
        druid:
          driver-class-name: com.mysql.cj.jdbc.Driver
          url: jdbc:mysql://localhost:3306/sky_takeout?serverTimezone=Asia/Shanghai&useUnicode=true&characterEncoding=utf8
          username: root
          password: 你本地的MySQL密码 # <--- 在这里修改
    ```
5. 找到 `sky-server` 下的 `SkyApplication.java`，点击绿色小箭头启动，不报错即证明环境通畅。

---

## 💻 二、 手把手写核心代码：员工登录

### 1. 编写 Mapper 接口方法
来到 `sky-server/src/main/java/com/sky/mapper/EmployeeMapper.java`。
我们需要根据前端传来的用户名去数据库查人：
```java
package com.sky.mapper;

import com.sky.entity.Employee;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface EmployeeMapper {
    @Select("SELECT * FROM employee WHERE username = #{username}")
    Employee getByUsername(String username);
}
```

### 2. 编写 Service 层逻辑
来到 `sky-server/.../service/impl/EmployeeServiceImpl.java`。
实现登录的业务逻辑（密码比对）：
```java
@Service
public class EmployeeServiceImpl implements EmployeeService {
    @Autowired
    private EmployeeMapper employeeMapper;

    @Override
    public Employee login(EmployeeLoginDTO employeeLoginDTO) {
        String username = employeeLoginDTO.getUsername();
        String password = employeeLoginDTO.getPassword();

        // 1. 根据用户名查询数据库
        Employee employee = employeeMapper.getByUsername(username);

        // 2. 各种校验
        if (employee == null) { throw new AccountNotFoundException("账号不存在"); }
        // 密码 MD5 加密后进行比对
        password = DigestUtils.md5DigestAsHex(password.getBytes());
        if (!password.equals(employee.getPassword())) { throw new PasswordErrorException("密码错误"); }
        if (employee.getStatus() == StatusConstant.DISABLE) { throw new AccountLockedException("账号被锁定"); }

        return employee;
    }
}
```

### 3. 编写 Controller 层暴露接口
来到 `sky-server/.../controller/admin/EmployeeController.java`。
```java
@RestController
@RequestMapping("/admin/employee")
@Slf4j
public class EmployeeController {
    @Autowired
    private EmployeeService employeeService;
    @Autowired
    private JwtProperties jwtProperties; // application.yml 里配置的 jwt 秘钥

    @PostMapping("/login")
    public Result<EmployeeLoginVO> login(@RequestBody EmployeeLoginDTO employeeLoginDTO) {
        log.info("员工登录：{}", employeeLoginDTO);
        Employee employee = employeeService.login(employeeLoginDTO);

        // 登录成功后，生成 JWT 令牌
        Map<String, Object> claims = new HashMap<>();
        claims.put(JwtClaimsConstant.EMP_ID, employee.getId());
        String token = JwtUtil.createJWT(
                jwtProperties.getAdminSecretKey(),
                jwtProperties.getAdminTtl(),
                claims);

        // 封装返回给前端的 VO 对象
        EmployeeLoginVO employeeLoginVO = EmployeeLoginVO.builder()
                .id(employee.getId())
                .userName(employee.getUsername())
                .name(employee.getName())
                .token(token)
                .build();

        return Result.success(employeeLoginVO);
    }
}
```

---

## 🛡️ 三、 设置请求安全关卡：JWT 拦截器

如果不做拦截，任何人用 Postman 都能访问 `/admin/employee/page` 查我们的员工。所以我们要搭建拦截器。

### 1. 编写拦截器类
来到 `sky-server/.../interceptor/JwtTokenAdminInterceptor.java`。
```java
@Component
public class JwtTokenAdminInterceptor implements HandlerInterceptor {
    @Autowired
    private JwtProperties jwtProperties;

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        // 如果不是拦截到动态方法的执行（比如静态文件），直接放行
        if (!(handler instanceof HandlerMethod)) { return true; }

        // 1. 获取请求头的 Token
        String token = request.getHeader(jwtProperties.getAdminTokenName());
        try {
            // 2. 解析 Token
            Claims claims = JwtUtil.parseJWT(jwtProperties.getAdminSecretKey(), token);
            Long empId = Long.valueOf(claims.get(JwtClaimsConstant.EMP_ID).toString());
            
            // 3. ✨ 核心动作：将员工ID存入当前线程专属内存！
            BaseContext.setCurrentId(empId);
            
            return true; // 验证通过，放行请求
        } catch (Exception ex) {
            response.setStatus(401); // 401 Unauthorized
            return false;
        }
    }
}
```

### 2. 注册拦截器
光有类不行，还要在 Spring MVC 里注册。来到 `sky-server/.../config/WebMvcConfiguration.java`。
```java
@Configuration
public class WebMvcConfiguration extends WebMvcConfigurationSupport {
    @Autowired
    private JwtTokenAdminInterceptor jwtTokenAdminInterceptor;

    @Override
    protected void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(jwtTokenAdminInterceptor)
                .addPathPatterns("/admin/**") // 拦截所有 admin 下的请求
                .excludePathPatterns("/admin/employee/login"); // 但排除登录接口！
    }
}
```

---

## 🏃 四、 测试与调试
1. 点击工程右上角的 `Debug` 虫子图标启动 `SkyApplication`。
2. 打开 `localhost:8080/doc.html` (Swagger/Knife4j 接口文档页面)。
3. 找到员工登录接口，在参数框输入（直接使用数据库第一条记录）：
   ```json
   {
     "username": "admin",
     "password": "123456" 
   }
   ```
   *(注意：如果 Day 0 初始化的密码是别的，请查阅 `employee` 表或重新改写)*
4. 点击发送，如果返回包含 `token` 的 JSON，恭喜你，后端的大门成功向你敞开！
