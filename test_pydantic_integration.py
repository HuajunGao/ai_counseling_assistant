"""Test script for Pydantic integration in LLM JSON parsing."""

import json
from core.llm_models import VisitorProfile, ProofreadResult, ProofreadMessage
from core.llm_engine import clean_json_response
from pydantic import ValidationError


def test_clean_json_response():
    """Test the clean_json_response function."""
    print("=" * 60)
    print("测试 1: clean_json_response() 函数")
    print("=" * 60)
    
    # Test case 1: JSON with ```json wrapper
    test_cases = [
        {
            "name": "带 ```json 标记",
            "input": '```json\n{"description": "测试", "personal_info": {"age": null}}\n```',
            "expected": '{"description": "测试", "personal_info": {"age": null}}'
        },
        {
            "name": "带 ``` 标记",
            "input": '```\n{"description": "测试", "personal_info": {"age": null}}\n```',
            "expected": '{"description": "测试", "personal_info": {"age": null}}'
        },
        {
            "name": "纯 JSON（无标记）",
            "input": '{"description": "测试", "personal_info": {"age": null}}',
            "expected": '{"description": "测试", "personal_info": {"age": null}}'
        },
        {
            "name": "前缀 ```json",
            "input": '```json\n{"description": "测试", "personal_info": {"age": null}}',
            "expected": '{"description": "测试", "personal_info": {"age": null}}'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        result = clean_json_response(case["input"])
        # Normalize whitespace for comparison
        result_normalized = result.strip()
        expected_normalized = case["expected"].strip()
        
        status = "✅ 通过" if result_normalized == expected_normalized else "❌ 失败"
        print(f"\n测试 1.{i} - {case['name']}: {status}")
        if result_normalized != expected_normalized:
            print(f"  期望: {expected_normalized}")
            print(f"  实际: {result_normalized}")


def test_visitor_profile_validation():
    """Test VisitorProfile Pydantic model validation."""
    print("\n" + "=" * 60)
    print("测试 2: VisitorProfile Pydantic 验证")
    print("=" * 60)
    
    # Test case 1: Valid profile
    print("\n测试 2.1 - 有效的档案数据:")
    valid_json = '''
    {
        "description": "一位面临职场压力的互联网从业者",
        "personal_info": {
            "age": "28岁",
            "gender": "女",
            "occupation": "产品经理",
            "background": "工作压力大"
        }
    }
    '''
    try:
        profile = VisitorProfile.model_validate_json(valid_json)
        print(f"✅ 验证通过")
        print(f"  描述: {profile.description}")
        print(f"  年龄: {profile.personal_info.age}")
        print(f"  性别: {profile.personal_info.gender}")
    except ValidationError as e:
        print(f"❌ 验证失败: {e}")
    
    # Test case 2: Profile with null values
    print("\n测试 2.2 - 包含 null 值的档案:")
    partial_json = '''
    {
        "description": "一位来访者",
        "personal_info": {
            "age": null,
            "gender": null,
            "occupation": null,
            "background": ""
        }
    }
    '''
    try:
        profile = VisitorProfile.model_validate_json(partial_json)
        print(f"✅ 验证通过")
        print(f"  描述: {profile.description}")
        print(f"  年龄: {profile.personal_info.age}")
    except ValidationError as e:
        print(f"❌ 验证失败: {e}")
    
    # Test case 3: Invalid profile (missing required field)
    print("\n测试 2.3 - 缺少必填字段:")
    invalid_json = '''
    {
        "personal_info": {
            "age": "28岁"
        }
    }
    '''
    try:
        profile = VisitorProfile.model_validate_json(invalid_json)
        print(f"❌ 应该失败但通过了")
    except ValidationError as e:
        print(f"✅ 正确捕获验证错误")
        print(f"  错误: {e.error_count()} 个字段错误")


def test_proofread_result_validation():
    """Test ProofreadResult Pydantic model validation."""
    print("\n" + "=" * 60)
    print("测试 3: ProofreadResult Pydantic 验证")
    print("=" * 60)
    
    # Test case 1: Valid proofread result
    print("\n测试 3.1 - 有效的纠错结果:")
    valid_json = '''
    {
        "messages": [
            {"role": "倾诉者", "text": "我今天很开心"},
            {"role": "倾听者", "text": "能说说是什么让你开心吗？"}
        ]
    }
    '''
    try:
        result = ProofreadResult.model_validate_json(valid_json)
        print(f"✅ 验证通过")
        print(f"  消息数量: {len(result.messages)}")
        for i, msg in enumerate(result.messages, 1):
            print(f"  消息 {i}: {msg.role} - {msg.text}")
    except ValidationError as e:
        print(f"❌ 验证失败: {e}")
    
    # Test case 2: With merged_from field
    print("\n测试 3.2 - 包含合并信息:")
    merged_json = '''
    {
        "messages": [
            {"role": "倾诉者", "text": "我今天出去玩了", "merged_from": [0, 1]}
        ]
    }
    '''
    try:
        result = ProofreadResult.model_validate_json(merged_json)
        print(f"✅ 验证通过")
        msg = result.messages[0]
        print(f"  文本: {msg.text}")
        print(f"  合并自: {msg.merged_from}")
    except ValidationError as e:
        print(f"❌ 验证失败: {e}")


def test_end_to_end_with_markdown():
    """Test end-to-end flow with markdown-wrapped JSON."""
    print("\n" + "=" * 60)
    print("测试 4: 端到端测试（模拟 LLM 返回带 markdown 的 JSON）")
    print("=" * 60)
    
    # Simulate LLM response with markdown
    llm_response = '''```json
{
    "description": "一位寻求帮助的来访者",
    "personal_info": {
        "age": "30多岁",
        "gender": "男",
        "occupation": "软件工程师",
        "background": "工作和生活压力较大"
    }
}
```'''
    
    print("\n模拟 LLM 响应:")
    print(llm_response[:100] + "...")
    
    try:
        # Step 1: Clean
        cleaned = clean_json_response(llm_response)
        print(f"\n✅ 步骤 1: 清理成功")
        
        # Step 2: Validate with Pydantic
        profile = VisitorProfile.model_validate_json(cleaned)
        print(f"✅ 步骤 2: Pydantic 验证成功")
        
        # Step 3: Convert to dict
        profile_dict = profile.model_dump()
        print(f"✅ 步骤 3: 转换为字典成功")
        print(f"\n最终结果:")
        print(f"  描述: {profile_dict['description']}")
        print(f"  职业: {profile_dict['personal_info']['occupation']}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Pydantic 集成测试")
    print("=" * 60)
    
    try:
        test_clean_json_response()
        test_visitor_profile_validation()
        test_proofread_result_validation()
        test_end_to_end_with_markdown()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
