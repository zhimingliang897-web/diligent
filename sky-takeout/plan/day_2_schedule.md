# Day 2: 菜品与套餐的多表事务挑战与 AOP 代码简化

## 🎯 学习目标
我们将使用 Spring AOP（面向切面编程）的魔力，彻底告别每次写 `setCreateTime` 的苦恼。同时，我们将攻克项目中第一个难点——包含多对多、一对多关系的“新增菜品与口味”，并实现阿里云 OSS 文件的上传。

---

## ✂️ 一、 AOP 黑科技：公共字段自动填充

每天写四遍 `setTime`, `setUser` 太折磨人了。我们来写一个“切面”帮我们在所有插入数据库的 SQL 执行前把时间塞进去。

### 1. 编写自定义注解 `@AutoFill`
来到 `sky-server/src/main/java/com/sky/annotation/AutoFill.java`。
```java
package com.sky.annotation;

import com.sky.enumeration.OperationType;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * 自定义注解：用于标识需要进行公共字段自动填充的方法
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface AutoFill {
    // 指定操作类型: UPDATE 还是 INSERT
    OperationType value();
}
```

### 2. 编写切面类 `AutoFillAspect.java`
来到 `sky-server/src/main/java/com/sky/aspect/AutoFillAspect.java`。
这一步是面试中常常让你手写的最经典的一段“反射动态注入”。
```java
@Aspect
@Component
@Slf4j
public class AutoFillAspect {

    // 1. 定义切入点：拦截 com.sky.mapper 包下所有带 @AutoFill 注解的方法！
    @Pointcut("execution(* com.sky.mapper.*.*(..)) && @annotation(com.sky.annotation.AutoFill)")
    public void autoFillPointCut(){}

    // 2. 前置通知：在 Mapper 执行 insert / update 之前，先来我这里执行
    @Before("autoFillPointCut()")
    public void autoFill(JoinPoint joinPoint){
        log.info("==> 开始进行公共字段的自动填充(AOP 切面)");

        // (1). 获取当前被拦截方法的具体注解（看是插入还是更新）
        MethodSignature signature = (MethodSignature) joinPoint.getSignature();
        AutoFill autoFill = signature.getMethod().getAnnotation(AutoFill.class);
        OperationType operationType = autoFill.value();

        // (2). 获取当前被拦截方法的参数（即要装进数据库的那个实体对象，比如 Employee）
        Object[] args = joinPoint.getArgs();
        if(args == null || args.length == 0){ return; }
        Object entity = args[0];

        // (3). 准备公共数据：时间与当前登录人的 ID (从 Day 1 的 ThreadLocal 中取)
        LocalDateTime now = LocalDateTime.now();
        Long currentId = BaseContext.getCurrentId();

        // (4). 神奇的 Java 反射：利用属性对应的 setter 方法把值塞进去
        try {
            // 通过反射拿到 setCreateTime 这个方法
            Method setCreateTime = entity.getClass().getDeclaredMethod(AutoFillConstant.SET_CREATE_TIME, LocalDateTime.class);
            Method setCreateUser = entity.getClass().getDeclaredMethod(AutoFillConstant.SET_CREATE_USER, Long.class);
            Method setUpdateTime = entity.getClass().getDeclaredMethod(AutoFillConstant.SET_UPDATE_TIME, LocalDateTime.class);
            Method setUpdateUser = entity.getClass().getDeclaredMethod(AutoFillConstant.SET_UPDATE_USER, Long.class);

            // 如果是新增操作，全盘塞入
            if(operationType == OperationType.INSERT){
                setCreateTime.invoke(entity, now);
                setCreateUser.invoke(entity, currentId);
                setUpdateTime.invoke(entity, now);
                setUpdateUser.invoke(entity, currentId);
            } else if(operationType == OperationType.UPDATE){
                // 更新的时候，创建时间不能变！
                setUpdateTime.invoke(entity, now);
                setUpdateUser.invoke(entity, currentId);
            }
        } catch (Exception e) {
            log.error("公共字段自动填充发生了异常！", e);
        }
    }
}
```

### 3. 如何使用它？
在你之前写的 `EmployeeMapper` 的 `insert` 或 `update` 的方法上加上：
```java
@AutoFill(value = OperationType.INSERT)
@Insert("insert into employee(name, username, password, ...) values (...) ")
void insert(Employee employee);
```
然后去跑一遍代码，就算你不 `setUpdateTime()`，数据库表里也照样有完美的时间和用户记录！

---

## ☁️ 二、 文件上传：阿里云 OSS 接入

你要给菜品配图。图片存在哪里？不要存在自己的 C 盘下，去买一个最便宜的阿里云对象存储（OSS）。

### 1. 配置 Yaml 信息
在 `application.yml` 和 `application-dev.yml` 配置相关参数：
```yaml
sky:
  alioss:
    endpoint: oss-cn-hangzhou.aliyuncs.com # 杭州节点
    access-key-id: LTAI5t你的KeyID     # 去 RAM 访问控制台拿
    access-key-secret: 你的KeySecret
    bucket-name: sky-takeout-xxx       # 你的桶名称
```

### 2. 编写上传 Controller
在 `sky-server` 的 `CommonController.java`：
```java
@RestController
@RequestMapping("/admin/common")
@Slf4j
public class CommonController {

    @Autowired
    private AliOssUtil aliOssUtil; // 苍穹源码包里 common 模块已提供

    @PostMapping("/upload")
    public Result<String> upload(MultipartFile file){
        log.info("接收到了菜品图片上传请求，大小：{}", file.getSize());

        try {
            // 获取原始带后缀的文件名：例如 foo.jpg
            String originalFilename = file.getOriginalFilename();
            String extension = originalFilename.substring(originalFilename.lastIndexOf("."));
            
            // 为了防止同名图片覆盖，用 UUID 做前缀重新命名：3fbd1...4a2.jpg
            String objectName = UUID.randomUUID().toString() + extension;

            // 调用 Util 工具类传给阿里云，返回值是一个能够在浏览器访问的 URL
            String filePath = aliOssUtil.upload(file.getBytes(), objectName);
            
            // 把外网能访问的 URL 丢给前端
            return Result.success(filePath); 
        } catch (IOException e) {
            log.error("哎呀，上传文件失败：{}", e);
        }
        return Result.error(MessageConstant.UPLOAD_FAILED);
    }
}
```

---

## 🔗 三、 “一损俱损”的跨表事务：新增菜品与口味

老板要新增一道【毛血旺 (Dish)】，它有两种【辣度 (Dish_Flavor：微辣、中辣)】。
如果你只存了菜，口味发生异常没存进去，系统就产生了“脏数据”。我们要靠 `@Transactional` 捆绑它们！

### 1. 控制器接收复合数据
因为前端会传来菜品+口味列表（List），我们需要 DTO 来接收：
```java
@PostMapping
public Result save(@RequestBody DishDTO dishDTO) {
    dishService.saveWithFlavor(dishDTO);
    return Result.success();
}
```

### 2. Service 层的核心爆发点！
打起精神，这段 `saveWithFlavor` 是整天最硬核的地方。
来到 `DishServiceImpl.java`。
```java
@Service
@Slf4j
public class DishServiceImpl implements DishService {

    @Autowired
    private DishMapper dishMapper;
    @Autowired
    private DishFlavorMapper dishFlavorMapper;

    /**
     * 新增菜品并且同时插入关联的口味数据！
     * 一重报错，全盘滚回原点！
     */
    @Transactional
    public void saveWithFlavor(DishDTO dishDTO) {

        // 1. 拆包，先把包含基本信息的 Dish 丢给菜品表（dish）
        Dish dish = new Dish();
        BeanUtils.copyProperties(dishDTO, dish);

        // 调用 Mapper 插入。注意！！！ 这里一旦执行完，由于 MyBatis 的特性，
        // dish.getId() 里已经被塞进了刚生成的自增主键。我们接下来要靠它！
        dishMapper.insert(dish);

        Long dishId = dish.getId(); // <--- 关键点：拿到热腾腾新出炉的 ID

        // 2. 处理口味数据，把它们丢进口味表（dish_flavor）
        List<DishFlavor> flavors = dishDTO.getFlavors();
        if (flavors != null && flavors.size() > 0) {
            // 给每个孤儿口味，找到它们的“亲爹”（刚才的 dishId）
            flavors.forEach(dishFlavor -> {
                dishFlavor.setDishId(dishId);
            });

            // 批量一次性强插，效率最高！
            dishFlavorMapper.insertBatch(flavors);
        }
    }
}
```

### 3. XML 里神秘的返回自增主键语法
在 `DishMapper.xml` 里，怎么让 `dishMapper.insert(dish)` 一执行完就能立刻把主键赋值给 `dish` 对象？
**关键在标签里加参数：`useGeneratedKeys="true" keyProperty="id"`**
```xml
<insert id="insert" useGeneratedKeys="true" keyProperty="id">
    insert into dish (name, category_id, price, image, description, create_time, update_time, create_user, update_user, status)
    values
    (#{name}, #{categoryId}, #{price}, #{image}, #{description}, #{createTime}, #{updateTime}, #{createUser}, #{updateUser}, #{status})
</insert>
```
