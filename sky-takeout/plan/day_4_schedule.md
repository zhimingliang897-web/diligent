# Day 4: 终局之战：订单状态全生命周期与 WebSocket 实录

## 🎯 学习目标
打响苍穹外卖难度最高的一战。你将亲手编写如何让用户的购物车变成一笔真实的锁库订单（防超卖基础），如何利用定时任务（Spring Task）清理那些“占着茅坑不拉屎”的僵尸订单，以及如何让商家的网页像微信聊天一样“叮”一声收到新业务（WebSocket）。

---

## 🛍️ 一、 用户下单的惊险之旅

电商的心脏在于“下单”。这个动作不仅仅是在表里写条数据那么简单。

### 1. 从购物车到订单表
在 `sky-server/.../service/impl/OrderServiceImpl.java` 中。
**重点警告：下单必须开启事务 `@Transactional`！**
由于我们要动两张表：主表 `orders` 和明细表 `order_detail`。

```java
@Service
@Slf4j
public class OrderServiceImpl implements OrderService {

    @Autowired
    private OrderMapper orderMapper;
    @Autowired
    private OrderDetailMapper orderDetailMapper;
    @Autowired
    private ShoppingCartMapper shoppingCartMapper;
    @Autowired
    private AddressBookMapper addressBookMapper;

    @Transactional // <--- 护身符
    public OrderSubmitVO submitOrder(OrdersSubmitDTO ordersSubmitDTO) {
        
        // 1. 获取当前点外卖的人的 ID
        Long userId = BaseContext.getCurrentId();

        // 2. 防呆校验：地址簿空不空？购物车到底有没有菜？
        AddressBook addressBook = addressBookMapper.getById(ordersSubmitDTO.getAddressBookId());
        if(addressBook == null){ throw new AddressBookBusinessException("兄弟，地址没选"); }
        // 查购物车
        ShoppingCart shoppingCart = new ShoppingCart();
        shoppingCart.setUserId(userId);
        List<ShoppingCart> shoppingCartList = shoppingCartMapper.list(shoppingCart);
        if(shoppingCartList == null || shoppingCartList.size() == 0) { throw new ShoppingCartBusinessException("购物车是空的"); }

        // 3. 构建主订单对象 Orders
        Orders orders = new Orders();
        BeanUtils.copyProperties(ordersSubmitDTO, orders);
        orders.setOrderTime(LocalDateTime.now());
        orders.setPayStatus(Orders.UN_PAID); // 刚下完单绝对是未支付(0)
        orders.setStatus(Orders.PENDING_PAY); // 处于待付款状态(1)
        orders.setNumber(String.valueOf(System.currentTimeMillis())); // 演示用时间戳当粗糙订单号
        orders.setPhone(addressBook.getPhone());
        orders.setConsignee(addressBook.getConsignee());
        orders.setUserId(userId);
        
        // 【核心】插入 order 表，并在 Mapper 层使用 useGeneratedKeys 拿到生成的 getId()！
        orderMapper.insert(orders); 

        // 4. 把购物车里的每一道菜，转换成订单明细 OrderDetail，并绑上主订单的 ID
        List<OrderDetail> orderDetailList = new ArrayList<>();
        for (ShoppingCart cart : shoppingCartList) {
            OrderDetail orderDetail = new OrderDetail(); // 明细对象
            BeanUtils.copyProperties(cart, orderDetail);
            orderDetail.setOrderId(orders.getId()); // 【绑死父订单】
            orderDetailList.add(orderDetail);
        }
        
        // 批量全插发明细表
        orderDetailMapper.insertBatch(orderDetailList);

        // 5. 过河拆桥：买完了，清空该用户的整个购物车！
        shoppingCartMapper.deleteByUserId(userId);

        // 返回给前端一个 VO 对象
        return OrderSubmitVO.builder().id(orders.getId()).orderNumber(orders.getNumber()).orderAmount(orders.getAmount()).orderTime(orders.getOrderTime()).build();
    }
}
```

---

## 💸 二、 支付回调与状态机流转

这里讲沙箱模拟思路：前端调用微信唤起密码锁，微信扣钱后，并不直接告诉前端，而是给咱们 Java 后台发一条秘密的 HTTP 请求。这叫**微信支付异步回调**。

```java
@RestController
@RequestMapping("/notify")
public class NotifyController {
    @Autowired
    private OrderService orderService;

    /** 微信的服务器会主动 POST 这个接口 */
    @PostMapping("/paySuccess")
    public void paySuccessNotify(@RequestBody String xmlData){ // 微信喜欢发 XML
        // 1. 解析出微信告诉我们的订单号 outTradeNo
        // 2. 去数据库修改这笔订单的状态：从“待付款”(1) 变成 “待接单”(2)
        // orderService.paySuccess(outTradeNo);

        // 3. 给马化腾回一封信（也是XML）：<return_code>SUCCESS</return_code>
        // 否则微信会以为断网了不停地给你发通知。
    }
}
```

---

## ⏳ 三、 僵尸订单清道夫：Spring Task 定时任务

如果那个老哥下完单就去睡觉了没付钱，库房的鸭血就会一直被锁在冰柜里？不行！超过 15 分钟不给钱的订单，全部物理取消。

我们在 `sky-server/.../task/OrderTask.java` 编写定时巡逻逻辑。
*注意：引导类上必须有 `@EnableScheduling` 才能开启扫描！*

```java
@Component
@Slf4j
public class OrderTask {

    @Autowired
    private OrderMapper orderMapper;

    /** 处理未支付一直占坑的订单 */
    @Scheduled(cron = "0 * * * * ? ") // 每 1 分钟，Spring 会自己悄悄调用这个方法一次
    public void processTimeoutOrder(){
        log.info("==> 开始巡视：有没有超过15分钟还没给钱的订单？");

        // 我们要干掉的订单的边界时间：当前真实时间往前拨 15 分钟
        LocalDateTime time = LocalDateTime.now().plusMinutes(-15);

        // 捞出来：select * from orders where status = 1 (待支付) and order_time < 那个15分钟前的时间
        List<Orders> orderList = orderMapper.getByStatusAndOrderTimeLT(Orders.PENDING_PAY, time);

        if(orderList != null && orderList.size() > 0){
            for (Orders orders : orderList) {
                // 无情更改为 6 (已取消)
                orders.setStatus(Orders.CANCELLED);
                orders.setCancelReason("订单超时未支付，自动系统取消");
                orders.setCancelTime(LocalDateTime.now());
                orderMapper.update(orders);
            }
        }
    }
}
```

---

## 🔊 四、 WebSockets 的语音播报魔法

平时前后端是 `前端问，后端答` 的 HTTP 短连接。可商家在看店时，不可能每秒钟狂刷新一次网页吧？我们要建立 WebSocket 长连接通道，让后端 **主动** 喊商家端：“来单了！”

### 1. 建立长链接端点
在 `sky-server/.../websocket/WebSocketServer.java`。苍穹源码给了一个大框架。
原理是使用 `@ServerEndpoint("/ws/{sid}")` 注解，把所有连上来的商家客户端（Session）保管在一个全局大 `Map` 集合里。

### 2. 扣动扳机的时刻
什么时候喊商家端？自然是 **《第二阶段：微信支付成功回调》** 刚完成状态变更的那一刻！

回到刚才提到的接到微信付款成功的那个 Service 方法里：
```java
public void paySuccess(String outTradeNo) {
    // 省略：把订单改成【待接单】的逻辑

    // 接下来，通过 WebSocket 群发一个 JSON 消息弹窗！
    Map map = new HashMap();
    map.put("type", 1); // 1 意思是：有新订单啦
    map.put("orderId", orders.getId()); // 传个ID，让前端做个点击跳转
    map.put("content", "您有新的外卖订单，号：" + outTradeNo);

    // 把大 Map 转成 JSON 字符串，调用 WebSocketServer 的群发工具方法
    String json = JSON.toJSONString(map);
    webSocketServer.sendToAllClient(json);
}
```
网页端接收到这串 JSON 就会唤起预置好的 MP3 组件：“您有新的外卖订单，请注意查收——”
