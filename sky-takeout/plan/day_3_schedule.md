# Day 3: 小程序登录与高并发缓存 (Redis 与双写一致)

## 🎯 学习目标
前端切入 C 端界面！今天我们将使用微信官方提供的免密登录，通过 Java 后端发请求拿 OpenID 并完成用户的静默注册。重点内容：如何用 Redis 缓解数据库的高并发读写。

---

## 📱 一、 C 端破冰：微信免密登录

让用户每次在小程序输入账号密码是不可能的，我们依赖微信。

### 1. 配置基础参数
你要有个注册好的微信小程序 AppID 和 Secret，填在 `application-dev.yml` 里：
```yaml
sky:
  wechat:
    appid: wx你申请的AppID
    secret: 你的秘钥内容
```

### 2. HttpClient 后端去向微信换取 OpenID
写一个能发送 HTTP 请求的客户端代码，由于涉及复杂的授权流，原版教程里封装了 `HttpClientUtil`。来 `sky-server/.../service/impl/UserServiceImpl.java` 里写登录核心逻辑：

```java
@Service
@Slf4j
public class UserServiceImpl implements UserService {

    // 微信官方固定的接口地址
    public static final String WX_LOGINUrl = "https://api.weixin.qq.com/sns/jscode2session";

    @Autowired
    private WeChatProperties weChatProperties;
    @Autowired
    private UserMapper userMapper;

    /** 微信登录静默流程 */
    public User login(UserLoginDTO userLoginDTO) {
        
        // 1. 调用微信接口服务，拿着前端给的 Code 获取这名用户独一无二的 OpenID
        Map<String, String> map = new HashMap<>();
        map.put("appid", weChatProperties.getAppid());
        map.put("secret", weChatProperties.getSecret());
        map.put("js_code", userLoginDTO.getCode());
        map.put("grant_type", "authorization_code");
        
        // 利用封装好的 HttpClient 发送 GET 请求
        String json = HttpClientUtil.doGet(WX_LOGINUrl, map);
        
        // 把微信服务器回的一团字符串 JSON 转成对象
        JSONObject jsonObject = JSON.parseObject(json);
        String openid = jsonObject.getString("openid");

        // 如果连 OpeniD 都没获取到，说明用户造假，或者断网
        if(openid == null){
            throw new LoginFailedException(MessageConstant.LOGIN_FAILED);
        }

        // 2. 用 OpenID 查这人我们自家数据库（User 表）里有没有
        User user = userMapper.getByOpenid(openid);
        
        // 3. 没查到说明是彻头彻尾的新用户
        if(user == null){
            user = User.builder()
                    .openid(openid)
                    .createTime(LocalDateTime.now())
                    .build();
            userMapper.insert(user); // 插入表，顺带生成了 ID
        }

        // 4. 返回查到的（或刚注册的）老哥对象，Controller 里拿到去签发咱们自己的 C 端 JWT Token
        return user;
    }
}
```

---

## 🚀 二、 内存提速王：Redis 与 Spring Data 底层
我们要把频繁访问的展示列表存到内存（Redis）里。

### 1. 解决序列化乱码（必须配置）
不做这步，存进 Redis 里的中文和对象全是一堆乱码。
写一个配置类，给 `RedisTemplate` 换上能读懂 JSON 对象的 String 序列化器。
`sky-server/.../config/RedisConfiguration.java`：
```java
@Configuration
public class RedisConfiguration {
    @Bean
    public RedisTemplate redisTemplate(RedisConnectionFactory redisConnectionFactory){
        RedisTemplate redisTemplate = new RedisTemplate();
        redisTemplate.setConnectionFactory(redisConnectionFactory);
        redisTemplate.setKeySerializer(new StringRedisSerializer());
        // 值的序列化用别的方法处理，比如 FastJson，这里简述。
        return redisTemplate;
    }
}
```

### 2. 把菜品查库操作前面挡上一层缓存！
去 `sky-server/.../controller/user/DishController.java`，也就是普通打工人点外卖的那个查询接口：

```java
@RestController("userDishController")
@RequestMapping("/user/dish")
@Slf4j
public class DishController {
    @Autowired
    private DishService dishService;
    @Autowired
    private RedisTemplate redisTemplate;

    /** 让前端调用能快到飞起来的方法 */
    @GetMapping("/list")
    public Result<List<DishVO>> list(Long categoryId) {
        // 1. 构造保存在 Redis 里的 Key：例如 dish_1001 代表 1001 分类下的所有起售菜品
        String key = "dish_" + categoryId;

        // 2. 【核心动作】先问问 Redis 大哥，兜里有没有我要的数据？
        List<DishVO> list = (List<DishVO>) redisTemplate.opsForValue().get(key);
        if(list != null && list.size() > 0){
             // 爽歪歪，Redis 直接有数据，丢给前端，MySQL 可以接着睡大觉了！
             return Result.success(list);
        }

        // 3. 实在没办法，Redis 兜底翻车了，只能老老实实去查笨重的 MySQL
        Dish dish = new Dish();
        dish.setCategoryId(categoryId);
        dish.setStatus(StatusConstant.ENABLE); // 只能展示起售中的数据
        
        list = dishService.listWithFlavor(dish);

        // 4. 【核心动作】查都查出来了，顺手丢回 Redis 大哥兜里一份，方便下一个顾客不用查这破库了
        redisTemplate.opsForValue().set(key, list);

        return Result.success(list);
    }
}
```

### 3. 一面必问：缓存双写与数据一致性处理
老板在后台偷偷涨价了！这时候，数据库 `dish` 表是新版，可 Redis 大哥兜里的菜还是旧版！这叫“脏数据”。咱们必须清理。

去**管理员管理后台的控制器 (`admin/DishController`)** 里面，找到那些带有 `修改`、`起售/停售` 和 `删除` 的方法，在最后硬加一段清缓存的代码：
```java
// 清理 Redis 缓存：如果是老板改了某个分类，直接把 dish_1001 整个 Key 删掉
// 下一个顾客在买饭的时候发现 Redis 是空的，就会自动触发去查库的流程！
private void cleanCache(String pattern) {
    Set keys = redisTemplate.keys(pattern); // 通配符找键: "dish_*" 全部拿下
    redisTemplate.delete(keys);
}
```

---

## 🛒 三、 初步购物车开发
C端加购物车，实际上是一张临时账单 `shopping_cart`。

这部分不用贴长篇大论，直接教最硬核的核心业务 `ShoppingServiceImpl.java`: 
当你收到前端丢来的一个 `dishId=10L, flavor="微辣"` 准备加入购物车时：
1. 拼凑 `select *` 语句去表里找当前 `userId` 这个人。
2. 发现表里有一模一样的这道菜，调用 `update shopping_cart set number = number + 1 where id = {那个查出来的ID}`。
3. 如果返回的是 NULL，说明这哥们是第一次点毛血旺，老老实实生成一个对象，插进去 `insert into shopping_cart (user_id, dish_id, number) values (1, 10, 1)`。

*(进阶探讨：企业里购物车都是塞到 Redis Hash 里的，咱们先用 MySQL 打通链路)*。
